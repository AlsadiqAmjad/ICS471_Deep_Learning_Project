from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, TensorDataset

from milestone3.data import build_augmentation, build_transform_pair, make_loaders
from milestone3.models import SkinLesionCNN
from milestone3.summary import cnn_block_summary
from milestone3.training import EarlyStopping, fit, load_checkpoint, save_checkpoint
from milestone3.transfer import build_transfer_model, extract_features, weight_layers


def _trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNNTests(unittest.TestCase):
    def test_output_shape_and_conv_count(self) -> None:
        model = SkinLesionCNN(num_classes=7, base_filters=8)
        model.eval()
        self.assertEqual(tuple(model(torch.randn(2, 3, 128, 128)).shape), (2, 7))
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.assertGreaterEqual(len(convs), 4)

    def test_rejects_fewer_than_four_convs(self) -> None:
        with self.assertRaises(ValueError):
            SkinLesionCNN(num_blocks=3, convs_per_block=1)

    def test_block_summary_shapes_and_parameter_total(self) -> None:
        model = SkinLesionCNN(base_filters=8)
        summary = cnn_block_summary(model, (3, 128, 128))
        self.assertEqual(summary.iloc[1]["output_shape"], "8 x 64 x 64")
        self.assertEqual(summary.iloc[4]["output_shape"], "64 x 8 x 8")
        self.assertEqual(int(summary["trainable_parameters"].sum()), _trainable(model))


class DataTests(unittest.TestCase):
    def test_augmentation_keeps_shape_and_dtype(self) -> None:
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        augmented = build_augmentation(64)(image)
        self.assertEqual(augmented.shape, image.shape)
        self.assertEqual(augmented.dtype, torch.uint8)

    def test_eval_transform_is_deterministic(self) -> None:
        train_t, eval_t = build_transform_pair(32, [0.5] * 3, [0.25] * 3)
        image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        self.assertTrue(torch.equal(eval_t(image), eval_t(image)))
        self.assertEqual(eval_t(image).dtype, torch.float32)

    def test_augmentation_only_on_training_loader(self) -> None:
        manifest = pd.DataFrame({
            "cache_index": range(8), "dx": ["a", "b"] * 4,
            "split": ["train"] * 4 + ["validation"] * 4,
        })
        cache = torch.randint(0, 256, (8, 3, 16, 16), dtype=torch.uint8)
        train_t, eval_t = build_transform_pair(16, [0.5] * 3, [0.25] * 3)
        loaders = make_loaders(manifest, cache, {"a": 0, "b": 1}, train_t, eval_t, 4, 0, 1, False)
        self.assertIs(loaders["train"].dataset.transform, train_t)
        self.assertIs(loaders["validation"].dataset.transform, eval_t)


class EarlyStoppingTests(unittest.TestCase):
    def test_modes(self) -> None:
        loss = EarlyStopping("validation_loss")
        acc = EarlyStopping("train_accuracy")
        self.assertTrue(loss.is_improvement(0.5, 0.6))
        self.assertFalse(loss.is_improvement(0.6, 0.5))
        self.assertTrue(acc.is_improvement(0.7, 0.6))
        with self.assertRaises(ValueError):
            EarlyStopping("test_accuracy")

    def test_fit_stops_restores_best_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 3, 16, 16)
        y = torch.randint(0, 3, (64,))
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        model = SkinLesionCNN(num_classes=3, base_filters=4, num_blocks=2, convs_per_block=2)
        # A zero learning rate never improves, so training must stop after patience epochs.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        result = fit(model, loader, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"),
                     max_epochs=10, early_stopping=EarlyStopping("validation_loss", patience=2),
                     use_amp=False, verbose=False)
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.stopped_epoch, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, result, {"note": "test"})
            clone = SkinLesionCNN(num_classes=3, base_filters=4, num_blocks=2, convs_per_block=2)
            checkpoint = load_checkpoint(path, clone)
            self.assertEqual(checkpoint["metadata"]["note"], "test")
            for (_, a), (_, b) in zip(model.state_dict().items(), clone.state_dict().items()):
                self.assertTrue(torch.equal(a, b))


    def test_plateau_scheduler_reduces_learning_rate(self) -> None:
        torch.manual_seed(0)
        loader = DataLoader(TensorDataset(torch.randn(32, 3, 16, 16), torch.randint(0, 3, (32,))), batch_size=16)
        model = SkinLesionCNN(num_classes=3, base_filters=4, num_blocks=2, convs_per_block=2)
        # A negligible learning rate leaves the validation loss flat, so the plateau scheduler must decay it.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-12)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, eps=0.0)
        result = fit(model, loader, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"),
                     max_epochs=5, early_stopping=EarlyStopping("validation_loss", patience=10),
                     scheduler=scheduler, use_amp=False, verbose=False)
        self.assertEqual(result.history["learning_rate"].iloc[0], 1e-12)
        self.assertLess(result.history["learning_rate"].iloc[-1], 1e-12)


class TransferTests(unittest.TestCase):
    def test_strategies_freeze_correct_layers(self) -> None:
        for backbone in ("resnet50", "mobilenet_v3_large", "vgg19"):
            head = build_transfer_model(backbone, "head", pretrained=False)
            last3 = build_transfer_model(backbone, "last3", pretrained=False)
            extended = build_transfer_model(backbone, "extended", pretrained=False)
            self.assertEqual(_trainable(head), head.final_linear.weight.numel() + 7)
            unfrozen = [layer for _, layer in weight_layers(last3.network)
                        if any(p.requires_grad for p in layer.parameters())]
            self.assertEqual(len(unfrozen), 3, backbone)
            self.assertGreater(_trainable(last3), _trainable(head))
            self.assertGreater(_trainable(extended), _trainable(head))
            self.assertEqual(extended.final_linear.out_features, 7)

    def test_frozen_batchnorm_stays_in_eval_mode(self) -> None:
        model = build_transfer_model("resnet50", "extended", pretrained=False)
        model.train()
        for module in model.network.modules():
            if isinstance(module, _BatchNorm):
                frozen = not any(p.requires_grad for p in module.parameters())
                self.assertEqual(module.training, not frozen)

    def test_feature_extraction_returns_penultimate_features(self) -> None:
        model = build_transfer_model("mobilenet_v3_large", "head", pretrained=False)
        loader = DataLoader(TensorDataset(torch.randn(5, 3, 64, 64), torch.arange(5)), batch_size=2)
        features, labels = extract_features(model, loader, torch.device("cpu"))
        self.assertEqual(features.shape, (5, 1280))
        np.testing.assert_array_equal(labels, np.arange(5))


if __name__ == "__main__":
    unittest.main()
