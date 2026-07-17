from __future__ import annotations

import unittest

import torch

from mdm_ddpo.runtime import (
    canonicalize_text_embeddings,
    collate_text_embeddings,
    split_text_embeddings,
)


class TextEmbeddingCacheTest(unittest.TestCase):
    def test_canonicalization_makes_permuted_bert_output_contiguous(self):
        source = torch.randn(2, 5, 3).permute(1, 0, 2)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        encoded, canonical_mask = canonicalize_text_embeddings((source, mask))
        self.assertTrue(encoded.is_contiguous())
        self.assertTrue(canonical_mask.is_contiguous())
        torch.testing.assert_close(encoded, source)

    def test_bert_embeddings_and_padding_round_trip_exactly(self):
        encoded = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
        padding_mask = torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, False, False],
            ]
        )
        entries = split_text_embeddings((encoded, padding_mask))
        rebuilt, rebuilt_mask = collate_text_embeddings(
            entries,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(rebuilt, encoded)
        torch.testing.assert_close(rebuilt_mask, padding_mask)


if __name__ == "__main__":
    unittest.main()
