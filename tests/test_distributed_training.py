import os
import unittest

import torch
import torch.distributed as dist

from trainer import Trainer, accumulation_windows
from utils import get_training_batch_info, init_device_seed
from vq import VectorQuantizer, sinkhorn_algorithm


class BatchConfigurationTest(unittest.TestCase):
    def test_paper_batch_size(self):
        single_gpu = get_training_batch_info(256, 1, 1)
        low_memory = get_training_batch_info(32, 2, 4)
        self.assertEqual(single_gpu['effective_batch_size'], 256)
        self.assertEqual(low_memory['effective_batch_size'], 256)

    def test_accumulation_is_global(self):
        config = get_training_batch_info(64, 2, 4)
        self.assertEqual(config['effective_batch_size'], 512)

    def test_accumulation_windows_keep_final_partial_batch(self):
        windows = list(accumulation_windows(range(5), 2))
        self.assertEqual(windows, [[0, 1], [2, 3], [4]])

    def test_process_seed_depends_on_rank(self):
        self.assertEqual(init_device_seed(2020, 0), 2020)
        self.assertEqual(init_device_seed(2020, 1), 2021)

    def test_preserved_forward_batch_rejects_qs_loss(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = {
            'auto_lambda_mode': 'fixed',
            'use_simple_uncertainty_loss': True,
        }
        with self.assertRaisesRegex(ValueError, 'qs_loss_weight=0'):
            trainer._validate_preserved_forward_batch_config({'qs_loss': 0.1})


class DistributedQuantizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if int(os.environ.get('WORLD_SIZE', '1')) > 1 and not dist.is_initialized():
            dist.init_process_group(backend='gloo')

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        if not dist.is_initialized():
            self.skipTest('Run with torchrun --nproc_per_node=2 to test distributed synchronization')

    @staticmethod
    def _make_quantizer():
        config = {
            'e_dim': 2,
            'beta': 0.25,
            'kmeans_init': False,
            'kmeans_iters': 10,
            'sk_iters': 20,
            'sync_quantizer_stats': True,
        }
        quantizer = VectorQuantizer(config, n_e=4, dist='l2', sk_epsilon=0.05)
        with torch.no_grad():
            quantizer.embedding.weight.copy_(torch.tensor([
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]))
        quantizer.train()
        return quantizer

    def test_sinkhorn_matches_global_batch(self):
        rank = dist.get_rank()
        local_inputs = torch.tensor([
            [-0.9, -0.8],
            [-0.8, 0.9],
        ]) if rank == 0 else torch.tensor([
            [0.8, -0.9],
            [0.9, 0.8],
        ])

        quantizer = self._make_quantizer()
        _, _, local_indices, _, _, _, _, _ = quantizer(local_inputs, use_sinkhorn=True)

        all_inputs = [torch.empty_like(local_inputs) for _ in range(dist.get_world_size())]
        dist.all_gather(all_inputs, local_inputs)
        global_inputs = torch.cat(all_inputs, dim=0)
        codebook = quantizer.embedding.weight
        distances = (
            torch.sum(global_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(codebook ** 2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(global_inputs, codebook.t())
        )
        middle = (distances.max() + distances.min()) / 2
        amplitude = (distances.max() - middle).clamp_min(1e-5)
        assignments = sinkhorn_algorithm(
            ((distances - middle) / amplitude).double(), 0.05, 20
        ).argmax(dim=-1)
        expected = assignments[rank * local_inputs.shape[0]:(rank + 1) * local_inputs.shape[0]]
        self.assertTrue(torch.equal(local_indices, expected))

    def test_balance_mean_uses_all_processes(self):
        rank = dist.get_rank()
        probabilities = torch.tensor([
            [0.8, 0.2],
            [0.6, 0.4],
        ], requires_grad=True) if rank == 0 else torch.tensor([
            [0.1, 0.9],
            [0.3, 0.7],
        ], requires_grad=True)

        quantizer = self._make_quantizer()
        global_mean = quantizer._distributed_batch_mean(probabilities)
        self.assertTrue(torch.allclose(global_mean, torch.tensor([0.45, 0.55])))

        global_mean.sum().backward()
        self.assertTrue(torch.allclose(probabilities.grad, torch.ones_like(probabilities) * 0.5))


if __name__ == '__main__':
    unittest.main()
