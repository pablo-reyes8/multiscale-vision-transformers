import torch
from torch.utils.data import DataLoader, TensorDataset

from famous_vits import ViTOrchestrator

TINY_VIT = {"embed_dim": 24, "depth": 1, "num_heads": 3}


def tiny_loader():
    images = torch.randn(4, 1, 32, 32)
    targets = torch.tensor([0, 1, 0, 1])
    return images, DataLoader(TensorDataset(images, targets), batch_size=2)


def test_orchestrator_trains_predicts_and_analyzes():
    images, loader = tiny_loader()
    vit = ViTOrchestrator(
        "vit",
        num_classes=2,
        in_chans=1,
        optimizer="adam",
        device="cpu",
        **TINY_VIT,
    )
    history = vit.fit(loader, epochs=1, val_loader=loader, verbose=False)
    assert len(history["train_loss"]) == 1
    assert vit.predict(images).shape == (4, 2)
    assert vit.extract_features(images).shape == (4, 24)
    assert vit.evaluate(loader)["top1"] >= 0

    analysis = vit.analyze(loader, class_names=["zero", "one"], num_bins=4)
    assert analysis["confusion_matrix"].shape == (2, 2)
    assert set(analysis["report"]["per_class"]) == {"zero", "one"}
    assert 0 <= analysis["calibration"]["ece"] <= 1


def test_checkpoint_reconstructs_orchestrator(tmp_path):
    checkpoint = tmp_path / "vit.pt"
    vit = ViTOrchestrator("vit", num_classes=3, in_chans=1, device="cpu", **TINY_VIT)
    vit.save(checkpoint)

    restored = ViTOrchestrator.from_checkpoint(checkpoint, device="cpu", load_optimizer=True)
    assert restored.summary()["architecture"] == "vit"
    assert restored.summary()["input_channels"] == 1
    assert restored.predict(torch.randn(1, 1, 32, 32)).shape == (1, 3)


def test_explainability_utilities_are_available_for_registered_vits():
    vit = ViTOrchestrator("vit", num_classes=3, in_chans=1, device="cpu", **TINY_VIT)
    images = torch.randn(1, 1, 32, 32)

    cam, cam_classes = vit.grad_cam(images)
    occlusion, occlusion_classes = vit.occlusion_sensitivity(
        images,
        patch_size=16,
        stride=16,
    )

    assert cam.shape == (1, 32, 32)
    assert occlusion.shape == (1, 32, 32)
    assert cam_classes.shape == occlusion_classes.shape == (1,)
