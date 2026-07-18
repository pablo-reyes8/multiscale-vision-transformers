import torch
import torch.nn as nn

from training.one_epoch import evaluate_one_epoch
from validation.utils import load_model_state


def test_model(
    model: nn.Module,
    checkpoint_path: str,
    test_loader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    label_smoothing: float = 0.0):

    load_model_state(model, checkpoint_path, device=device)
    model.to(device)

    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        use_amp=use_amp,
        label_smoothing=label_smoothing)

    print(
        f"[Test] Loss: {test_loss:.4f} | "
        f"Top-1: {test_metrics['top1']:.2f}% | "
        f"Top-3: {test_metrics['top3']:.2f}% | "
        f"Top-5: {test_metrics['top5']:.2f}%")

    return test_loss, test_metrics
