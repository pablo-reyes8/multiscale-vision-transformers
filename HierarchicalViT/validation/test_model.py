import torch
import torch.nn as nn
from training.train_one_epoch import *


def test_model(
    model: nn.Module,
    checkpoint_path: str,
    test_loader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,):

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        use_amp=use_amp)

    print(
        f"[Test] Loss: {test_loss:.4f} | "
        f"Top-1: {test_metrics['top1']:.2f}% | "
        f"Top-3: {test_metrics['top3']:.2f}% | "
        f"Top-5: {test_metrics['top5']:.2f}%")

    return test_loss, test_metrics

