import os
import random
import tempfile
import unittest

import numpy as np
import torch

from trainer import (
    Trainer,
    capture_rng_state,
    restore_rng_state,
    resumable_config_fingerprint,
)
from vq import RQVAE


class _SingleProcessAccelerator:
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        return None


def _make_trainer(root):
    trainer = Trainer.__new__(Trainer)
    trainer.config = {
        "dataset": "beauty",
        "seed": 2020,
        "data_path": "/data/readonly",
        "epochs": 8,
        "batch_size": 2,
        "resume_from": None,
        "save_path": str(root),
        "stop_after_epoch": 0,
    }
    trainer.save_path = str(root)
    trainer.manifest_path = os.path.join(root, "manifest.json")
    trainer.device = torch.device("cpu")
    trainer.accelerator = _SingleProcessAccelerator()
    trainer.model_rec = torch.nn.Linear(3, 2)
    trainer.model_id = torch.nn.Linear(2, 2)
    trainer.rec_optimizer = torch.optim.AdamW(
        trainer.model_rec.parameters(), lr=1e-2
    )
    trainer.id_optimizer = torch.optim.AdamW(
        trainer.model_id.parameters(), lr=2e-2
    )
    trainer.rec_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.rec_optimizer, lambda step: 0.95 ** step
    )
    trainer.id_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.id_optimizer, lambda step: 0.9 ** step
    )
    trainer.all_item_code = torch.tensor([[-1, -1], [0, 1], [1, 0]])
    trainer.global_step = 3
    trainer.best_score = 0.25
    trainer.best_result = {"ndcg@10": 0.25}
    trainer.best_epoch = 0
    trainer.best_ckpt = os.path.join(root, "0.pt")
    trainer.last_validation_metrics = {"ndcg@10": 0.25}
    trainer.last_codebook_stats = {"sinkhorn": [{"used": 2}]}
    trainer.last_assignment_stats = [{"entropy": 0.5}]
    trainer.last_rec_gradient_report = {"codebooks": [0.1, 0.2]}
    trainer.valid_metric = "ndcg@10"
    trainer.log = lambda *args, **kwargs: None
    return trainer


def _optimization_step(trainer):
    trainer.rec_optimizer.zero_grad(set_to_none=True)
    trainer.id_optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(4, 3)
    hidden = trainer.model_rec(inputs)
    output = trainer.model_id(hidden)
    loss = output.square().mean()
    loss.backward()
    trainer.rec_optimizer.step()
    trainer.id_optimizer.step()
    trainer.rec_lr_scheduler.step()
    trainer.id_lr_scheduler.step()
    trainer.global_step += 1
    return output.detach().clone()


def _tiny_rq_config(assignment_mode):
    return {
        "layers": [6],
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
        "sync_quantizer_stats": False,
        "use_adaptive_selection": False,
        "use_learnable_sigma_gumbel": "sdud" in assignment_mode,
        "use_simple_uncertainty_loss": True,
        "initial_std": 0.1,
        "sigma_lambda": 1.7,
        "gumbel_noise_scale": 0.1,
        "hot_threshold_ratio": 1.1,
        "usage_momentum": 0.99,
    }


def _make_ud_trainer(root, assignment_mode):
    trainer = _make_trainer(root)
    trainer.config.update(
        {
            "training_mode": assignment_mode,
            "assignment_temperature": 2.0,
            "gumbel_noise_scale": 0.1,
            "hot_threshold_ratio": 1.1,
            "usage_momentum": 0.99,
            "use_learnable_sigma_gumbel": "sdud" in assignment_mode,
            "use_simple_uncertainty_loss": True,
            "initial_std": 0.1,
            "sigma_lambda": 1.7,
        }
    )
    trainer.model_rec = torch.nn.Linear(4, 4)
    trainer.model_id = RQVAE(_tiny_rq_config(assignment_mode), in_dim=4)
    trainer.rec_optimizer = torch.optim.AdamW(
        trainer.model_rec.parameters(), lr=1e-2
    )
    trainer.id_optimizer = torch.optim.AdamW(
        trainer.model_id.parameters(), lr=2e-3
    )
    trainer.rec_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.rec_optimizer, lambda step: 0.95 ** step
    )
    trainer.id_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.id_optimizer, lambda step: 0.9 ** step
    )
    trainer.all_item_code = torch.tensor(
        [[-1, -1, -1, -1], [0, 1, 2, 0], [1, 2, 3, 0]]
    )
    return trainer


def _ud_optimization_step(trainer, assignment_mode):
    trainer.rec_optimizer.zero_grad(set_to_none=True)
    trainer.id_optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(6, 4)
    tokenizer_output = trainer.model_id(
        inputs,
        return_structured=True,
        assignment_temperature=2.0,
        assignment_mode=assignment_mode,
        assignment_forward="hard_st",
    )
    output = trainer.model_rec(tokenizer_output.quantized)
    loss = output.square().mean() + tokenizer_output.vq_loss
    loss.backward()
    trainer.rec_optimizer.step()
    trainer.id_optimizer.step()
    trainer.rec_lr_scheduler.step()
    trainer.id_lr_scheduler.step()
    trainer.global_step += 1
    return output.detach().clone()


def _assert_nested_equal(test_case, actual, expected, path="state"):
    if torch.is_tensor(expected):
        test_case.assertTrue(torch.equal(actual, expected), path)
    elif isinstance(expected, dict):
        test_case.assertEqual(set(actual), set(expected), path)
        for key in expected:
            _assert_nested_equal(
                test_case, actual[key], expected[key], f"{path}.{key}"
            )
    elif isinstance(expected, (list, tuple)):
        test_case.assertEqual(len(actual), len(expected), path)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(
                test_case, actual_item, expected_item, f"{path}[{index}]"
            )
    else:
        test_case.assertEqual(actual, expected, path)


class RngCheckpointTest(unittest.TestCase):
    def test_all_rng_streams_restore_exactly(self):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        state = capture_rng_state()
        expected = (
            random.random(),
            np.random.random(),
            torch.rand(5),
        )
        restore_rng_state(state)
        actual = (
            random.random(),
            np.random.random(),
            torch.rand(5),
        )
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_runtime_only_options_do_not_change_resume_fingerprint(self):
        first = {"epochs": 120, "lr": 1e-5, "stop_after_epoch": 3}
        second = {
            "epochs": 120,
            "lr": 1e-5,
            "stop_after_epoch": 30,
            "resume_from": "/tmp/state.resume",
        }
        self.assertEqual(
            resumable_config_fingerprint(first),
            resumable_config_fingerprint(second),
        )
        second["lr"] = 2e-5
        self.assertNotEqual(
            resumable_config_fingerprint(first),
            resumable_config_fingerprint(second),
        )


class FullTrainingStateTest(unittest.TestCase):
    def test_save_load_continuation_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            source = _make_trainer(temporary)
            _optimization_step(source)
            checkpoint = source._save_resume_checkpoint(
                epoch=0,
                cur_eval_step=2,
                legacy_checkpoint=os.path.join(temporary, "0.pt"),
            )

            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_output = _optimization_step(source)
            expected_rec = {
                name: tensor.detach().clone()
                for name, tensor in source.model_rec.state_dict().items()
            }
            expected_id = {
                name: tensor.detach().clone()
                for name, tensor in source.model_id.state_dict().items()
            }

            resumed = _make_trainer(os.path.join(temporary, "resumed"))
            next_epoch, cur_eval_step = resumed._load_resume_checkpoint(checkpoint)
            self.assertEqual(next_epoch, 1)
            self.assertEqual(cur_eval_step, 2)
            self.assertEqual(resumed.all_item_code.tolist(), source.all_item_code.tolist())
            self.assertEqual(resumed.last_assignment_stats, [{"entropy": 0.5}])
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            actual_output = _optimization_step(resumed)

            self.assertTrue(torch.equal(actual_output, expected_output))
            for name, tensor in resumed.model_rec.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected_rec[name]), name)
            for name, tensor in resumed.model_id.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected_id[name]), name)
            _assert_nested_equal(
                self,
                resumed.rec_optimizer.state_dict(),
                source.rec_optimizer.state_dict(),
            )
            _assert_nested_equal(
                self,
                resumed.id_optimizer.state_dict(),
                source.id_optimizer.state_dict(),
            )
            _assert_nested_equal(
                self,
                resumed.rec_lr_scheduler.state_dict(),
                source.rec_lr_scheduler.state_dict(),
            )
            _assert_nested_equal(
                self,
                resumed.id_lr_scheduler.state_dict(),
                source.id_lr_scheduler.state_dict(),
            )

    def test_config_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(23)
            source = _make_trainer(temporary)
            checkpoint = source._save_resume_checkpoint(
                epoch=0,
                cur_eval_step=0,
                legacy_checkpoint=os.path.join(temporary, "0.pt"),
            )
            resumed = _make_trainer(os.path.join(temporary, "resumed"))
            resumed.config["batch_size"] = 4
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                resumed._load_resume_checkpoint(checkpoint)

    def test_uncertainty_modes_restore_dynamic_state_and_next_step_exactly(self):
        for assignment_mode in (
            "true_e2e_frqud",
            "true_e2e_sdud",
            "true_e2e_sdud_frqud",
        ):
            with self.subTest(assignment_mode=assignment_mode):
                with tempfile.TemporaryDirectory() as temporary:
                    random.seed(31)
                    np.random.seed(31)
                    torch.manual_seed(31)
                    source = _make_ud_trainer(temporary, assignment_mode)
                    _ud_optimization_step(source, assignment_mode)
                    checkpoint = source._save_resume_checkpoint(
                        epoch=0,
                        cur_eval_step=1,
                        legacy_checkpoint=os.path.join(temporary, "0.pt"),
                    )
                    saved_id_state = {
                        name: tensor.detach().clone()
                        for name, tensor in source.model_id.state_dict().items()
                    }

                    expected_output = _ud_optimization_step(source, assignment_mode)
                    expected_rec_state = {
                        name: tensor.detach().clone()
                        for name, tensor in source.model_rec.state_dict().items()
                    }
                    expected_id_state = {
                        name: tensor.detach().clone()
                        for name, tensor in source.model_id.state_dict().items()
                    }

                    resumed = _make_ud_trainer(
                        os.path.join(temporary, "resumed"), assignment_mode
                    )
                    resumed._load_resume_checkpoint(checkpoint)
                    for name, tensor in resumed.model_id.state_dict().items():
                        self.assertTrue(torch.equal(tensor, saved_id_state[name]), name)

                    actual_output = _ud_optimization_step(resumed, assignment_mode)
                    self.assertTrue(torch.equal(actual_output, expected_output))
                    for name, tensor in resumed.model_rec.state_dict().items():
                        self.assertTrue(torch.equal(tensor, expected_rec_state[name]), name)
                    for name, tensor in resumed.model_id.state_dict().items():
                        self.assertTrue(torch.equal(tensor, expected_id_state[name]), name)
                    _assert_nested_equal(
                        self,
                        resumed.id_optimizer.state_dict(),
                        source.id_optimizer.state_dict(),
                    )
                    _assert_nested_equal(
                        self,
                        resumed.id_lr_scheduler.state_dict(),
                        source.id_lr_scheduler.state_dict(),
                    )


if __name__ == "__main__":
    unittest.main()
