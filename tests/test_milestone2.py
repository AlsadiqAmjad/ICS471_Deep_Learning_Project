from __future__ import annotations

import unittest

import pandas as pd
import torch

from milestone2.losses import FocalLoss
from milestone2.models import MLPClassifier
from milestone2.splitting import create_milestone2_splits, validate_split_manifest


class Milestone2Tests(unittest.TestCase):
    def test_grouped_split_has_no_leakage(self) -> None:
        rows = []
        classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        for label in classes:
            for group_number in range(30):
                lesion_id = f"{label}_{group_number}"
                for image_number in range(1 + int(group_number % 3 == 0)):
                    rows.append(
                        {
                            "lesion_id": lesion_id,
                            "image_id": f"{lesion_id}_{image_number}",
                            "dx": label,
                        }
                    )
        manifest = create_milestone2_splits(pd.DataFrame(rows))
        validate_split_manifest(manifest)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            left_groups = set(manifest.loc[manifest["split"] == left, "lesion_id"])
            right_groups = set(manifest.loc[manifest["split"] == right, "lesion_id"])
            self.assertFalse(left_groups & right_groups)

    def test_mlp_output_shape(self) -> None:
        model = MLPClassifier(3 * 16 * 16, [32, 16], 7, dropout=0.1)
        model.eval()
        output = model(torch.randn(8, 3, 16, 16))
        self.assertEqual(tuple(output.shape), (8, 7))

    def test_focal_loss_is_finite(self) -> None:
        criterion = FocalLoss(torch.ones(7), gamma=2.0)
        loss = criterion(torch.randn(8, 7), torch.arange(8) % 7)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
