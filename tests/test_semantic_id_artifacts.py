import unittest

import torch

from trainer import semantic_id_change_stats, semantic_id_map_sha256


class SemanticIdArtifactTest(unittest.TestCase):
    def test_change_stats_exclude_padding_row(self):
        previous = [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 0],
            [2, 2, 1],
        ]
        current = [
            [9, 9, 9],
            [0, 0, 0],
            [1, 2, 0],
            [2, 2, 0],
        ]
        stats = semantic_id_change_stats(previous, current, prefix_length=2)
        self.assertEqual(stats['item_count'], 3)
        self.assertEqual(stats['full']['changed_items'], 2)
        self.assertEqual(stats['prefix']['changed_items'], 1)
        self.assertEqual(stats['suffix']['changed_items'], 1)
        self.assertEqual(stats['per_level'][0]['changed_items'], 0)
        self.assertEqual(stats['per_level'][1]['changed_items'], 1)

    def test_map_hash_is_dtype_stable(self):
        codes = [[-1, -1], [0, 1], [2, 3]]
        self.assertEqual(
            semantic_id_map_sha256(codes),
            semantic_id_map_sha256(torch.tensor(codes, dtype=torch.int32)),
        )


if __name__ == '__main__':
    unittest.main()
