import unittest

import torch
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from model import Model
from trainer import (
    Trainer,
    cached_hard_straight_through,
    deduplicate_batch_items,
)
from vq import (
    RQVAE,
    VectorQuantizer,
    deterministic_straight_through,
    gumbel_straight_through,
)


def _tiny_config():
    return {
        "semantic_hidden_size": 6,
        "num_beams": 2,
        "layers": [8],
        "e_dim": 4,
        "num_emb_list": [4, 4, 4],
        "beta": 0.25,
        "alpha": 1.0,
        "vq_type": "vq",
        "dist": "l2",
        "kmeans_init": False,
        "kmeans_iters": 5,
        "dropout_prob": 0.0,
        "bn": False,
        "sk_epsilons": [0.0, 0.0, 0.0],
        "sk_iters": 5,
        "gumbel_tau": 2.0,
        "use_indicator_ste": True,
        "stop_gumbel_sampling_epoch": 0,
        "sync_quantizer_stats": False,
        "use_adaptive_selection": False,
        "use_learnable_sigma_gumbel": False,
        "gumbel_hard_switch_epoch": 0,
        "gumbel_noise_scale": 1.0,
        "usage_momentum": 1.0,
        "hot_threshold_ratio": 1.5,
        "use_simple_uncertainty_loss": True,
        "initial_std": 1.0,
        "sigma_lambda": 1.8,
    }


def _finite_nonzero(tensor):
    return (
        tensor is not None
        and torch.isfinite(tensor).all().item()
        and tensor.detach().float().norm().item() > 1e-10
    )


class StraightThroughAssignmentTest(unittest.TestCase):
    def test_hard_forward_and_soft_backward(self):
        logits = torch.tensor(
            [[0.2, 1.1, -0.4], [1.3, -0.2, 0.5]], requires_grad=True
        )
        soft, hard, straight_through, indices = deterministic_straight_through(
            logits, temperature=2.0
        )

        self.assertTrue(torch.equal(straight_through, hard))
        self.assertTrue(torch.equal(indices, logits.argmax(dim=-1)))
        self.assertTrue(torch.allclose(soft.sum(dim=-1), torch.ones(2)))

        weights = torch.tensor([[0.0, 1.0, 3.0], [2.0, -1.0, 0.5]])
        (straight_through * weights).sum().backward()
        self.assertTrue(_finite_nonzero(logits.grad))

    def test_detached_hard_assignment_is_negative_control(self):
        logits = torch.randn(3, 4, requires_grad=True)
        _, hard, _, _ = deterministic_straight_through(logits, temperature=2.0)
        loss = (hard.detach() * torch.randn_like(hard)).sum() + 0.0 * logits.sum()
        loss.backward()
        self.assertEqual(logits.grad.detach().float().norm().item(), 0.0)


class CachedHardStraightThroughTest(unittest.TestCase):
    def test_forward_is_exactly_cached_and_backward_is_current_soft(self):
        logits = torch.tensor(
            [
                [[0.2, 1.1, -0.4], [1.3, -0.2, 0.5]],
                [[0.7, -0.1, 0.3], [-0.4, 0.8, 0.2]],
            ],
            requires_grad=True,
        )
        current_soft = F.softmax(logits / 2.0, dim=-1)
        current_soft.retain_grad()
        cached_hard = torch.tensor([[2, 0], [1, 2]])

        sid_st = cached_hard_straight_through(current_soft, cached_hard)
        expected = F.one_hot(cached_hard, num_classes=3).to(logits.dtype)

        self.assertTrue(torch.equal(sid_st, expected))
        weights = torch.tensor(
            [
                [[0.1, 0.7, -0.2], [0.6, -0.5, 0.3]],
                [[-0.4, 0.2, 0.9], [0.8, 0.1, -0.3]],
            ]
        )
        (sid_st * weights).sum().backward()
        self.assertTrue(torch.equal(current_soft.grad, weights))
        self.assertTrue(_finite_nonzero(logits.grad))

    def test_rejects_padding_or_out_of_range_cached_ids(self):
        current_soft = torch.full((2, 3, 4), 0.25)
        with self.assertRaises(ValueError):
            cached_hard_straight_through(
                current_soft, torch.tensor([[0, 1, 2], [1, -1, 3]])
            )
        with self.assertRaises(ValueError):
            cached_hard_straight_through(
                current_soft, torch.tensor([[0, 1, 2], [1, 4, 3]])
            )


class GumbelStraightThroughAssignmentTest(unittest.TestCase):
    def test_hard_forward_and_soft_backward_reach_logits_and_codebook(self):
        logits = torch.tensor(
            [[0.2, 1.1, -0.4], [1.3, -0.2, 0.5]], requires_grad=True
        )
        codebook = torch.randn(3, 5, requires_grad=True)
        torch.manual_seed(11)
        output = gumbel_straight_through(
            logits, temperature=2.0, noise_scale=1.0, add_noise=True
        )

        self.assertTrue(
            torch.equal(output.straight_through_probabilities, output.hard_one_hot)
        )
        self.assertTrue(
            torch.equal(
                output.hard_indices,
                output.noisy_logits.argmax(dim=-1),
            )
        )
        quantized = output.straight_through_probabilities @ codebook
        quantized.square().sum().backward()
        self.assertTrue(_finite_nonzero(logits.grad))
        self.assertTrue(_finite_nonzero(codebook.grad))

    def test_seed_reproducibility_and_training_stochasticity(self):
        logits = torch.zeros(32, 8)
        torch.manual_seed(17)
        first = gumbel_straight_through(logits, 2.0, 1.0, add_noise=True)
        torch.manual_seed(17)
        second = gumbel_straight_through(logits, 2.0, 1.0, add_noise=True)
        torch.manual_seed(18)
        third = gumbel_straight_through(logits, 2.0, 1.0, add_noise=True)

        self.assertTrue(torch.equal(first.hard_indices, second.hard_indices))
        self.assertTrue(torch.equal(first.gumbel_noise, second.gumbel_noise))
        self.assertFalse(torch.equal(first.hard_indices, third.hard_indices))

    def test_zero_scale_and_evaluation_match_deterministic_st(self):
        logits = torch.randn(7, 5)
        deterministic = deterministic_straight_through(logits, temperature=2.0)
        zero_scale = gumbel_straight_through(
            logits, 2.0, 0.0, add_noise=True
        )
        evaluation = gumbel_straight_through(
            logits, 2.0, 3.0, add_noise=False
        )

        for output in (zero_scale, evaluation):
            self.assertTrue(torch.equal(output.soft_probabilities, deterministic[0]))
            self.assertTrue(torch.equal(output.hard_one_hot, deterministic[1]))
            self.assertTrue(
                torch.equal(output.straight_through_probabilities, deterministic[2])
            )
            self.assertTrue(torch.equal(output.hard_indices, deterministic[3]))

    def test_quantizer_supports_hard_st_and_soft_training_forward(self):
        config = _tiny_config()
        quantizer = VectorQuantizer(
            config, n_e=4, dist="l2", sk_epsilon=0.0
        )
        quantizer.train()
        inputs = torch.randn(6, 4)

        torch.manual_seed(29)
        hard_output = quantizer(
            inputs,
            tau=2.0,
            return_structured=True,
            assignment_mode="true_e2e_gumbel_fixed",
            assignment_forward="hard_st",
        )
        torch.manual_seed(29)
        soft_output = quantizer(
            inputs,
            tau=2.0,
            return_structured=True,
            assignment_mode="true_e2e_gumbel_fixed",
            assignment_forward="soft",
        )

        self.assertTrue(
            torch.equal(
                hard_output.forward_probabilities,
                hard_output.hard_one_hot,
            )
        )
        self.assertTrue(
            torch.equal(
                soft_output.forward_probabilities,
                soft_output.soft_probabilities,
            )
        )
        self.assertTrue(
            torch.allclose(
                hard_output.quantized_vector,
                hard_output.hard_one_hot @ quantizer.embedding.weight,
            )
        )
        self.assertTrue(
            torch.allclose(
                soft_output.quantized_vector,
                soft_output.soft_probabilities @ quantizer.embedding.weight,
            )
        )

    def test_structured_vq_loss_preserves_scalar_mean(self):
        tokenizer = RQVAE(_tiny_config(), in_dim=6)
        tokenizer.train()
        output = tokenizer(
            torch.randn(5, 6),
            return_structured=True,
            assignment_mode="true_e2e_gumbel_fixed",
        )
        self.assertEqual(tuple(output.vq_loss_per_item.shape), (5,))
        self.assertTrue(
            torch.allclose(output.vq_loss, output.vq_loss_per_item.mean())
        )
        for level in output.levels:
            self.assertEqual(tuple(level.vq_loss_per_item.shape), (5,))
            self.assertTrue(
                torch.allclose(level.vq_loss, level.vq_loss_per_item.mean())
            )


class UncertaintyAssignmentTest(unittest.TestCase):
    def _quantizer(self, learnable_sigma=False):
        config = _tiny_config()
        config["use_learnable_sigma_gumbel"] = learnable_sigma
        quantizer = VectorQuantizer(config, n_e=4, dist="l2", sk_epsilon=0.0)
        with torch.no_grad():
            quantizer.embedding.weight.copy_(
                torch.tensor(
                    [[-2.0, 0.0, 0.0, 0.0],
                     [0.0, -2.0, 0.0, 0.0],
                     [0.0, 0.0, -2.0, 0.0],
                     [0.0, 0.0, 0.0, -2.0]]
                )
            )
        quantizer.train()
        return quantizer

    def test_frqud_uses_per_code_ema_frequency(self):
        quantizer = self._quantizer()
        quantizer.code_usage_ema.copy_(torch.tensor([0.7, 0.1, 0.1, 0.1]))
        inputs = torch.tensor(
            [[-2.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]]
        )
        output = quantizer(
            inputs,
            tau=2.0,
            return_structured=True,
            assignment_mode="true_e2e_frqud",
        )

        self.assertTrue(torch.equal(output.stochastic_mask, torch.tensor([True, False])))
        self.assertTrue(torch.allclose(output.frequency_scores, torch.tensor([0.7, 0.1])))
        self.assertTrue(torch.equal(output.effective_noise_scale, torch.tensor([1.0, 0.0])))

    def test_combined_mode_applies_sdud_noise_only_to_hot_assignments(self):
        quantizer = self._quantizer(learnable_sigma=True)
        quantizer.code_usage_ema.copy_(torch.tensor([0.7, 0.1, 0.1, 0.1]))
        inputs = torch.tensor(
            [[-2.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]]
        )
        output = quantizer(
            inputs,
            tau=2.0,
            return_structured=True,
            assignment_mode="true_e2e_sdud_frqud",
        )

        self.assertTrue(output.noise_scale.requires_grad)
        self.assertTrue(torch.equal(output.stochastic_mask, torch.tensor([True, False])))
        self.assertTrue(
            torch.allclose(
                output.effective_noise_scale,
                torch.stack([output.noise_scale, output.noise_scale * 0]),
            )
        )

    def test_quantizer_evaluation_ignores_noise_and_is_deterministic(self):
        quantizer = self._quantizer()
        inputs = torch.randn(6, 4)
        quantizer.eval()
        torch.manual_seed(31)
        first = quantizer(
            inputs,
            tau=2.0,
            return_structured=True,
            assignment_mode="true_e2e_gumbel_fixed",
        )
        torch.manual_seed(32)
        second = quantizer(
            inputs,
            tau=2.0,
            return_structured=True,
            assignment_mode="true_e2e_gumbel_fixed",
        )

        self.assertTrue(torch.equal(first.hard_indices, second.hard_indices))
        self.assertTrue(
            torch.equal(
                first.hard_indices,
                first.assignment_logits.argmax(dim=-1),
            )
        )
        self.assertEqual(first.effective_noise_scale.sum().item(), 0.0)


class BatchDeduplicationTest(unittest.TestCase):
    def test_padding_and_repeated_items(self):
        history = torch.tensor([[1, 2, 0, 0], [2, 3, 1, 0]])
        mask = history.ne(0)
        targets = torch.tensor([[4], [2]])

        index = deduplicate_batch_items(history, mask, targets, padding_id=0)

        self.assertTrue(torch.equal(index.unique_item_ids, torch.tensor([1, 2, 3, 4])))
        self.assertTrue((index.history_inverse[~mask] == -1).all())
        self.assertEqual(index.history_inverse[0, 0], index.history_inverse[1, 2])
        self.assertEqual(index.history_inverse[0, 1], index.target_inverse[1, 0])
        self.assertNotIn(0, index.unique_item_ids.tolist())


class CheckpointSelectionTest(unittest.TestCase):
    def test_first_zero_metric_epoch_still_establishes_a_checkpoint(self):
        trainer = Trainer.__new__(Trainer)
        trainer.valid_metric = "ndcg@10"
        trainer.best_score = 0.0
        trainer.best_ckpt = None
        self.assertTrue(
            trainer._is_best_checkpoint_candidate({"ndcg@10": 0.0})
        )
        trainer.best_ckpt = "0.pt"
        self.assertFalse(
            trainer._is_best_checkpoint_candidate({"ndcg@10": 0.0})
        )


class TrueEndToEndGradientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.use_deterministic_algorithms(True)
        config = _tiny_config()
        t5_config = T5Config(
            num_layers=1,
            num_decoder_layers=1,
            d_model=8,
            d_ff=16,
            num_heads=2,
            d_kv=4,
            dropout_rate=0.0,
            vocab_size=1,
            pad_token_id=0,
            eos_token_id=4,
            decoder_start_token_id=0,
        )
        self.recommender = Model(
            config,
            T5ForConditionalGeneration(t5_config),
            n_items=7,
            code_length=4,
            code_number=4,
        )
        self.tokenizer = RQVAE(config, in_dim=6)
        self.recommender.semantic_embedding.requires_grad_(False)
        self.recommender.train()
        self.tokenizer.train()

        self.history = torch.tensor([[1, 2, 0], [3, 1, 2]])
        self.history_mask = self.history.ne(0)
        self.targets = torch.tensor([[4], [5]])
        self.semantic_by_item = torch.tensor(
            [
                [-1, -1, -1],
                [2, 0, 1],
                [1, 3, 0],
                [0, 2, 3],
                [3, 1, 2],
                [2, 3, 1],
                [1, 0, 2],
            ]
        )
        self.suffix_by_item = torch.tensor([-1, 0, 1, 0, 2, 3, 0])

    def _clear_gradients(self):
        self.recommender.zero_grad(set_to_none=True)
        self.tokenizer.zero_grad(set_to_none=True)

    def _forward(
        self,
        detach_history=False,
        detach_labels=False,
        detach_teacher=False,
        tokenizer=None,
        assignment_mode="true_e2e_plain",
        assignment_forward="hard_st",
    ):
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        index = deduplicate_batch_items(
            self.history, self.history_mask, self.targets, padding_id=0
        )
        item_embeddings = self.recommender.semantic_embedding(index.unique_item_ids)
        tokenizer_output = tokenizer(
            item_embeddings,
            current_epoch=0,
            global_step=0,
            return_structured=True,
            deterministic_st=assignment_mode == "true_e2e_plain",
            assignment_mode=assignment_mode,
            assignment_forward=assignment_forward,
        )
        for level in tokenizer_output.levels:
            level.assignment_logits.retain_grad()

        current_soft = torch.stack(
            [level.soft_probabilities for level in tokenizer_output.levels],
            dim=1,
        )
        unique_probabilities = cached_hard_straight_through(
            current_soft,
            self.semantic_by_item[index.unique_item_ids],
        )
        history_probabilities = unique_probabilities[
            index.history_inverse.clamp_min(0)
        ]
        target_probabilities = unique_probabilities[index.target_inverse].squeeze(1)

        history_for_model = (
            history_probabilities.detach() if detach_history else history_probabilities
        )
        labels_for_loss = (
            target_probabilities.detach() if detach_labels else target_probabilities
        )
        teacher_for_model = (
            target_probabilities.detach() if detach_teacher else target_probabilities
        )

        history_suffix = self.suffix_by_item[self.history].clamp_min(0)
        target_suffix = self.suffix_by_item[self.targets.squeeze(1)]
        inputs_embeds, expanded_mask = self.recommender.get_mixture_input_embeddings(
            history_for_model,
            history_suffix,
            self.history_mask,
        )
        decoder_inputs_embeds = self.recommender.get_differentiable_decoder_inputs(
            teacher_for_model
        )
        outputs = self.recommender(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )
        semantic_loss, suffix_loss = self.recommender.compute_differentiable_code_losses(
            outputs.logits, labels_for_loss, target_suffix
        )
        return tokenizer_output, semantic_loss, suffix_loss

    def _assert_tokenizer_parameter_gradients(self, tokenizer=None):
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        encoder_gradients = [
            parameter.grad for parameter in tokenizer.encoder.parameters()
        ]
        self.assertTrue(any(_finite_nonzero(gradient) for gradient in encoder_gradients))
        for quantizer in tokenizer.rq.vq_layers:
            self.assertTrue(_finite_nonzero(quantizer.embedding.weight.grad))

    def test_recommendation_semantic_loss_reaches_all_assignment_levels(self):
        output, semantic_loss, _ = self._forward()
        semantic_loss.backward()

        self._assert_tokenizer_parameter_gradients()
        for level in output.levels:
            self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_all_gumbel_modes_preserve_recommendation_only_gradient_contract(self):
        for assignment_mode in (
            "true_e2e_gumbel_fixed",
            "true_e2e_frqud",
            "true_e2e_sdud",
            "true_e2e_sdud_frqud",
        ):
            with self.subTest(assignment_mode=assignment_mode):
                config = _tiny_config()
                if "sdud" in assignment_mode:
                    config["use_learnable_sigma_gumbel"] = True
                tokenizer = RQVAE(config, in_dim=6)
                tokenizer.train()
                self.recommender.zero_grad(set_to_none=True)
                tokenizer.zero_grad(set_to_none=True)
                torch.manual_seed(23)
                output, semantic_loss, _ = self._forward(
                    tokenizer=tokenizer,
                    assignment_mode=assignment_mode,
                )
                semantic_loss.backward()
                self._assert_tokenizer_parameter_gradients(tokenizer)
                for level in output.levels:
                    self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_sdud_keeps_raw_loss_and_noise_path_reaches_sigma_and_codebooks(self):
        config = _tiny_config()
        config["use_learnable_sigma_gumbel"] = True
        config["initial_std"] = 1.0
        tokenizer = RQVAE(config, in_dim=6)
        tokenizer.train()
        for parameter in tokenizer.encoder.parameters():
            parameter.requires_grad_(False)

        self.recommender.zero_grad(set_to_none=True)
        tokenizer.zero_grad(set_to_none=True)
        torch.manual_seed(53)
        output, raw_semantic_loss, suffix_loss = self._forward(
            tokenizer=tokenizer,
            assignment_mode="true_e2e_sdud",
        )
        semantic_loss = Trainer._true_e2e_semantic_loss(raw_semantic_loss)
        self.assertIs(semantic_loss, raw_semantic_loss)

        recon_loss = F.mse_loss(output.quantized, output.latent)
        expected_total = raw_semantic_loss + suffix_loss + recon_loss + output.vq_loss
        actual_total = semantic_loss + suffix_loss + recon_loss + output.vq_loss
        self.assertTrue(torch.equal(actual_total, expected_total))

        semantic_loss.backward()
        for quantizer in tokenizer.rq.vq_layers:
            self.assertTrue(_finite_nonzero(quantizer.embedding.weight.grad))
            self.assertTrue(
                _finite_nonzero(quantizer.auto_sigma_module.sigma.grad)
            )

    def test_soft_forward_recommendation_loss_reaches_every_codebook(self):
        output, semantic_loss, _ = self._forward(
            assignment_mode="true_e2e_gumbel_fixed",
            assignment_forward="soft",
        )
        semantic_loss.backward()
        self._assert_tokenizer_parameter_gradients()
        for level in output.levels:
            self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_frozen_encoder_still_gets_recommendation_gradient_to_all_codebooks(self):
        for parameter in self.tokenizer.encoder.parameters():
            parameter.requires_grad_(False)
        output, semantic_loss, _ = self._forward(
            assignment_mode="true_e2e_gumbel_fixed"
        )
        semantic_loss.backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in self.tokenizer.encoder.parameters())
        )
        for quantizer in self.tokenizer.rq.vq_layers:
            self.assertTrue(_finite_nonzero(quantizer.embedding.weight.grad))
        for level in output.levels:
            self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_gumbel_detach_negative_control_removes_recommendation_gradient(self):
        _, semantic_loss, suffix_loss = self._forward(
            detach_history=True,
            detach_labels=True,
            detach_teacher=True,
            assignment_mode="true_e2e_gumbel_fixed",
        )
        (semantic_loss + suffix_loss).backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in self.tokenizer.parameters())
        )

    def test_gumbel_modes_preserve_individual_recommendation_paths(self):
        for assignment_mode in (
            "true_e2e_gumbel_fixed",
            "true_e2e_frqud",
            "true_e2e_sdud",
            "true_e2e_sdud_frqud",
        ):
            config = _tiny_config()
            if "sdud" in assignment_mode:
                config["use_learnable_sigma_gumbel"] = True
            for detached_paths in (
                {"detach_labels": True, "detach_teacher": True},
                {"detach_history": True, "detach_teacher": True},
                {"detach_history": True, "detach_labels": True},
            ):
                with self.subTest(
                    assignment_mode=assignment_mode,
                    detached_paths=detached_paths,
                ):
                    tokenizer = RQVAE(config, in_dim=6)
                    tokenizer.train()
                    self.recommender.zero_grad(set_to_none=True)
                    tokenizer.zero_grad(set_to_none=True)
                    torch.manual_seed(37)
                    output, semantic_loss, suffix_loss = self._forward(
                        tokenizer=tokenizer,
                        assignment_mode=assignment_mode,
                        **detached_paths,
                    )
                    (semantic_loss + suffix_loss).backward()
                    self._assert_tokenizer_parameter_gradients(tokenizer)
                    for level in output.levels:
                        self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_history_path_alone_reaches_tokenizer(self):
        _, semantic_loss, _ = self._forward(
            detach_labels=True, detach_teacher=True
        )
        semantic_loss.backward()
        self._assert_tokenizer_parameter_gradients()

    def test_target_soft_label_path_alone_reaches_tokenizer(self):
        output, semantic_loss, _ = self._forward(
            detach_history=True, detach_teacher=True
        )
        semantic_loss.backward()
        self._assert_tokenizer_parameter_gradients()
        for level in output.levels:
            self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_teacher_forcing_path_reaches_previous_assignments(self):
        output, semantic_loss, suffix_loss = self._forward(
            detach_history=True, detach_labels=True
        )
        (semantic_loss + suffix_loss).backward()
        for level in output.levels:
            self.assertTrue(_finite_nonzero(level.assignment_logits.grad))

    def test_detaching_all_assignments_removes_recommendation_gradient(self):
        _, semantic_loss, suffix_loss = self._forward(
            detach_history=True, detach_labels=True, detach_teacher=True
        )
        (semantic_loss + suffix_loss).backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in self.tokenizer.parameters())
        )

    def test_auxiliary_and_recommendation_gradients_are_attributed_separately(self):
        output, semantic_loss, _ = self._forward()
        semantic_loss.backward()
        recommendation_norm = self.tokenizer.rq.vq_layers[0].embedding.weight.grad.norm()
        self.assertGreater(recommendation_norm.item(), 1e-10)

        self._clear_gradients()
        output, _, _ = self._forward(
            detach_history=True, detach_labels=True, detach_teacher=True
        )
        auxiliary_loss = F.mse_loss(output.quantized, output.latent) + output.vq_loss
        auxiliary_loss.backward()
        auxiliary_norm = self.tokenizer.rq.vq_layers[0].embedding.weight.grad.norm()
        self.assertGreater(auxiliary_norm.item(), 1e-10)

    def test_suffix_is_a_hard_detached_boundary(self):
        _, _, suffix_loss = self._forward()
        target_suffix = self.suffix_by_item[self.targets.squeeze(1)]
        self.assertEqual(target_suffix.dtype, torch.long)
        self.assertFalse(target_suffix.requires_grad)
        self.assertTrue(torch.isfinite(suffix_loss))

    def test_true_e2e_keeps_legacy_fixed_suffix_vocabulary(self):
        config = _tiny_config()
        config["training_mode"] = "true_e2e_plain"
        t5_config = T5Config(
            num_layers=1,
            num_decoder_layers=1,
            d_model=8,
            d_ff=16,
            num_heads=2,
            d_kv=4,
            dropout_rate=0.0,
            vocab_size=1,
            pad_token_id=0,
            eos_token_id=4,
            decoder_start_token_id=0,
        )
        recommender = Model(
            config,
            T5ForConditionalGeneration(t5_config),
            n_items=11,
            code_length=4,
            code_number=4,
        )
        self.assertEqual(
            [embedding.num_embeddings for embedding in recommender.token_embeddings],
            [4, 4, 4, 4],
        )

        history_probabilities = F.one_hot(
            torch.tensor([[[0, 1, 2]]]), num_classes=4
        ).float()
        inputs_embeds, attention_mask = recommender.get_mixture_input_embeddings(
            history_probabilities,
            torch.tensor([[3]]),
            torch.ones(1, 1, dtype=torch.bool),
        )
        target_probabilities = F.one_hot(
            torch.tensor([[1, 2, 3]]), num_classes=4
        ).float()
        outputs = recommender(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=recommender.get_differentiable_decoder_inputs(
                target_probabilities
            ),
        )
        self.assertEqual(tuple(outputs.logits.shape), (1, 4, 4))
        self.assertEqual(tuple(outputs.suffix_logits.shape), (1, 4))
        semantic_loss, suffix_loss = recommender.compute_differentiable_code_losses(
            outputs.logits,
            target_probabilities,
            torch.tensor([3]),
            suffix_logits=outputs.suffix_logits,
        )
        self.assertTrue(torch.isfinite(semantic_loss + suffix_loss))

        recommender.eval()
        hard_history = torch.tensor([[1, 2, 3, 3]])
        hard_mask = torch.ones_like(hard_history, dtype=torch.bool)
        first = recommender.generate(
            hard_history.clone(), hard_mask, n_return_sequences=2
        )
        second = recommender.generate(
            hard_history.clone(), hard_mask, n_return_sequences=2
        )
        self.assertTrue(torch.equal(first, second))
        self.assertLess(first[:, :, -1].max().item(), 4)

    def test_hard_history_forward_matches_one_hot_mixture(self):
        hard_codes = torch.tensor(
            [[[1, 2, 3, 0], [2, 1, 0, 1], [-1, -1, -1, -1]],
             [[0, 1, 2, 0], [1, 0, 3, 0], [2, 2, 1, 1]]]
        )
        hard_mask = hard_codes[..., 0].ne(-1)
        hard_flat = hard_codes.reshape(2, -1)
        old_embeddings = self.recommender.get_input_embeddings(
            hard_flat.clone(), hard_flat.ne(-1)
        )
        probabilities = F.one_hot(
            hard_codes[..., :3].clamp_min(0), num_classes=4
        ).float()
        new_embeddings, new_mask = self.recommender.get_mixture_input_embeddings(
            probabilities, hard_codes[..., 3].clamp_min(0), hard_mask
        )
        self.assertTrue(torch.equal(new_mask, hard_flat.ne(-1)))
        self.assertTrue(torch.allclose(new_embeddings, old_embeddings))

    def test_baseline_logits_and_ce_match_one_hot_differentiable_forward(self):
        history_codes = torch.tensor(
            [[[1, 2, 3, 0], [2, 1, 0, 1]],
             [[0, 1, 2, 0], [1, 0, 3, 0]]]
        )
        target_codes = torch.tensor([[3, 2, 1, 0], [1, 3, 2, 1]])
        hard_history = history_codes.reshape(2, -1)
        hard_mask = hard_history.ne(-1)
        baseline = self.recommender(
            input_ids=hard_history.clone(),
            attention_mask=hard_mask,
            labels=target_codes,
        )
        baseline_loss = F.cross_entropy(
            baseline.logits.reshape(-1, 4), target_codes.reshape(-1)
        )

        history_probabilities = F.one_hot(
            history_codes[..., :3], num_classes=4
        ).float()
        target_probabilities = F.one_hot(
            target_codes[:, :3], num_classes=4
        ).float()
        inputs_embeds, expanded_mask = self.recommender.get_mixture_input_embeddings(
            history_probabilities, history_codes[..., 3], torch.ones(2, 2).bool()
        )
        decoder_inputs = self.recommender.get_differentiable_decoder_inputs(
            target_probabilities
        )
        differentiable = self.recommender(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_mask,
            decoder_inputs_embeds=decoder_inputs,
        )
        semantic_loss, suffix_loss = self.recommender.compute_differentiable_code_losses(
            differentiable.logits, target_probabilities, target_codes[:, 3]
        )
        self.assertTrue(torch.allclose(differentiable.logits, baseline.logits))
        self.assertTrue(torch.allclose(semantic_loss + suffix_loss, baseline_loss))

    def test_one_batch_overfits_with_recommendation_loss(self):
        optimizer = torch.optim.Adam(
            list(self.recommender.parameters()) + list(self.tokenizer.parameters()),
            lr=0.01,
        )
        losses = []
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            _, semantic_loss, suffix_loss = self._forward()
            loss = semantic_loss + suffix_loss
            losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        self.assertLess(losses[-1], losses[0] * 0.75)


if __name__ == "__main__":
    unittest.main()
