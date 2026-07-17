import copy
import csv
import inspect
import json
import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..data import (
    build_transforms as build_dataset_transforms,
)
from ..data import (
    get_classification_dataloaders,
    get_dataset_info,
)
from ..factory import create_model
from .presets import (
    ARENA_DEFAULTS,
    AUGMENT_PRESETS,
    GENERAL_TIPS,
    MODEL_PRESETS,
    MODEL_TIPS,
    available_model_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


@dataclass
class RunResult:
    model: str
    family: str
    params: int
    trainable_params: int
    best_epoch: int | None
    best_val_loss: float | None
    best_val_top1: float | None
    best_val_top3: float | None
    best_val_top5: float | None
    test_loss: float | None
    test_top1: float | None
    test_top3: float | None
    test_top5: float | None
    train_minutes: float
    checkpoint_path: str | None
    augment: str


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3, 5)):
    with torch.no_grad():
        max_k = min(max(ks), logits.shape[1])
        batch_size = targets.size(0)
        _, pred = torch.topk(logits, k=max_k, dim=1)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        out = {}
        for k in ks:
            out[k] = (
                100.0
                * correct[:, : min(k, logits.shape[1])].any(dim=1).float().sum().item()
                / batch_size
            )
        return out


def build_param_groups_no_wd(model: nn.Module, weight_decay: float):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in lower_name
            or name in {"cls_token", "pos_embed"}
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class WarmupCosineLR:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        step = self.step_num

        for idx, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx]
            if step <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * (step / self.warmup_steps)
            else:
                capped_step = min(step, self.total_steps)
                denom = max(1, self.total_steps - self.warmup_steps)
                progress = (capped_step - self.warmup_steps) / denom
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            group["lr"] = lr


def _cuda_dtype_supported(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    return dtype in (torch.bfloat16, torch.float16)


def make_grad_scaler(device: str = "cuda", enabled: bool = True):
    if not enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if len(sig.parameters) >= 1:
                return torch.amp.GradScaler(device if device in ("cuda", "cpu") else "cuda")
            return torch.amp.GradScaler()
        except Exception:
            pass

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()

    return None


@contextmanager
def autocast_ctx(
    device: str = "cuda", enabled: bool = True, dtype: str = "bf16", cache_enabled: bool = True
):
    if not enabled:
        with nullcontext():
            yield
        return

    if device.startswith("cuda"):
        want = DTYPE_MAP.get(dtype.lower(), torch.bfloat16)
        use = want if _cuda_dtype_supported(want) else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=use, cache_enabled=cache_enabled):
            yield
        return

    if device == "cpu":
        try:
            with torch.amp.autocast(
                device_type="cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled
            ):
                yield
        except Exception:
            with nullcontext():
                yield
        return

    with nullcontext():
        yield


def describe_models():
    lines = []
    for model_name in available_model_names():
        preset = MODEL_PRESETS[model_name]
        lines.append(f"{model_name}: {preset['family']} | {preset['description']}")
    return "\n".join(lines)


def describe_tips():
    lines = ["General tips:"]
    for tip in GENERAL_TIPS:
        lines.append(f"- {tip}")
    lines.append("")
    lines.append("Model-specific tips:")
    for model_name in available_model_names():
        lines.append(f"- {model_name}:")
        for tip in MODEL_TIPS.get(model_name, []):
            lines.append(f"  - {tip}")
    return "\n".join(lines)


def resolve_model_names(requested_models: list[str] | None):
    if not requested_models:
        return ["vit", "hierarchical_vit", "swin"]

    if len(requested_models) == 1 and requested_models[0] == "all":
        return available_model_names()

    unknown = [name for name in requested_models if name not in MODEL_PRESETS]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Available: {available_model_names()}")
    return requested_models


def validate_preset(model_name: str):
    preset = MODEL_PRESETS[model_name]
    config = preset["config"]
    builder = preset["builder"]

    if builder == "vit":
        if config["embed_dim"] % config["num_heads"] != 0:
            raise ValueError(f"{model_name}: embed_dim must be divisible by num_heads")
        return

    if builder == "hierarchical_vit":
        if not (len(config["embed_dims"]) == len(config["depths"]) == len(config["num_heads"])):
            raise ValueError(f"{model_name}: embed_dims, depths, and num_heads must align by stage")
        for dim, heads in zip(config["embed_dims"], config["num_heads"], strict=False):
            if dim % heads != 0:
                raise ValueError(
                    f"{model_name}: every stage dim must be divisible by its num_heads"
                )
        return

    if builder == "swin":
        if not (len(config["depths"]) == len(config["num_heads"]) == 4):
            raise ValueError(f"{model_name}: this Swin implementation expects four stages")
        dims = [
            config["embed_dim"],
            config["embed_dim"] * 2,
            config["embed_dim"] * 4,
            config["embed_dim"] * 8,
        ]
        for dim, heads in zip(dims, config["num_heads"], strict=False):
            if dim % heads != 0:
                raise ValueError(
                    f"{model_name}: every stage dim must be divisible by its num_heads"
                )
        return

    if builder == "maxvit":
        return

    if builder == "volo":
        if config.get("hierarchical", False):
            lengths = [
                len(config["dims"]),
                len(config["outlooker_depths"]),
                len(config["outlooker_heads_list"]),
                len(config["transformer_depths"]),
                len(config["transformer_heads_list"]),
            ]
            if len(set(lengths)) != 1:
                raise ValueError(f"{model_name}: hierarchical tuple lengths must align by stage")
            if config["pooling"] != "mean":
                raise ValueError(f"{model_name}: hierarchical VOLO should use pooling='mean'")
        else:
            if config["embed_dim"] % config["outlooker_heads"] != 0:
                raise ValueError(f"{model_name}: embed_dim must be divisible by outlooker_heads")
            if config["embed_dim"] % config["transformer_heads"] != 0:
                raise ValueError(f"{model_name}: embed_dim must be divisible by transformer_heads")
        return


def build_model(
    model_name: str,
    img_size: int = 32,
    num_classes: int = 100,
    in_chans: int = 3,
):
    validate_preset(model_name)
    model = create_model(
        model_name,
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
    )
    return model, model.model_config


def build_dry_run_report(
    model_names: list[str],
    img_size: int = 32,
    num_classes: int = 100,
    in_chans: int = 3,
):
    rows = []
    for model_name in model_names:
        model, resolved_config = build_model(
            model_name,
            img_size=img_size,
            num_classes=num_classes,
            in_chans=in_chans,
        )
        params, trainable = count_parameters(model)
        rows.append(
            {
                "model": model_name,
                "family": MODEL_PRESETS[model_name]["family"],
                "params": params,
                "trainable_params": trainable,
                "config": resolved_config,
            }
        )
    return rows


def build_transforms(augment: str, img_size: int = 32):
    """Compatibility wrapper for the original CIFAR-100 arena helper."""

    return build_dataset_transforms("cifar100", img_size=img_size, augment=augment)


def make_cifar100_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    img_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.1,
    augment: str = "matched",
    seed: int = 42,
    pin_memory: bool = True,
):
    """Compatibility wrapper backed by the packaged dataset registry."""

    train_loader, val_loader, test_loader = get_classification_dataloaders(
        dataset_name="cifar100",
        data_dir=data_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        img_size=img_size,
        num_workers=num_workers,
        val_split=val_split,
        augment=augment,
        seed=seed,
        pin_memory=pin_memory,
    )
    split_info = {
        "seed": seed,
        "val_split": val_split,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset) if val_loader is not None else 0,
        "test_size": len(test_loader.dataset),
        "augment": augment,
    }
    return train_loader, val_loader, test_loader, split_info


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    scaler=None,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    label_smoothing: float = 0.1,
    print_every: int = 100,
    max_batches: int | None = None,
):
    model.train().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_scaler = (scaler is not None) and use_amp and autocast_dtype.lower() in ("fp16", "float16")

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0
    start = time.time()

    for step, (images, targets) in enumerate(dataloader, start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = criterion(logits.float(), targets)

        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        total += batch_size
        accs = accuracy_topk(logits.detach(), targets, ks=(1, 3, 5))
        c1 += accs[1] * batch_size / 100.0
        c3 += accs[3] * batch_size / 100.0
        c5 += accs[5] * batch_size / 100.0

        if print_every and step % print_every == 0:
            elapsed = time.time() - start
            imgs_sec = total / max(elapsed, 1e-9)
            print(
                f"[train step {step}] "
                f"loss {running_loss / total:.4f} | "
                f"top1 {100.0 * c1 / total:.2f}% | "
                f"top3 {100.0 * c3 / total:.2f}% | "
                f"top5 {100.0 * c5 / total:.2f}% | "
                f"{imgs_sec:.1f} img/s"
            )

    avg_loss = running_loss / max(total, 1)
    metrics = {
        "top1": 100.0 * c1 / max(total, 1),
        "top3": 100.0 * c3 / max(total, 1),
        "top5": 100.0 * c5 / max(total, 1),
    }
    return avg_loss, metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    label_smoothing: float = 0.1,
    max_batches: int | None = None,
):
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    for step, (images, targets) in enumerate(dataloader, start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = criterion(logits.float(), targets)

        running_loss += loss.item() * batch_size
        total += batch_size
        accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
        c1 += accs[1] * batch_size / 100.0
        c3 += accs[3] * batch_size / 100.0
        c5 += accs[5] * batch_size / 100.0

    avg_loss = running_loss / max(total, 1)
    metrics = {
        "top1": 100.0 * c1 / max(total, 1),
        "top3": 100.0 * c3 / max(total, 1),
        "top5": 100.0 * c5 / max(total, 1),
    }
    return avg_loss, metrics


def train_model_for_arena(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    output_dir: Path,
    augment: str,
    epochs: int,
    device: str,
    lr: float,
    weight_decay: float,
    autocast_dtype: str,
    use_amp: bool,
    grad_clip_norm: float | None,
    warmup_ratio: float,
    min_lr: float,
    label_smoothing: float,
    print_every: int,
    max_train_batches: int | None,
    max_eval_batches: int | None,
):
    model.to(device)
    params, trainable = count_parameters(model)

    param_groups = build_param_groups_no_wd(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = (
        min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader)
    )
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr
    )

    scaler = None
    if use_amp and autocast_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)

    history = []
    best_epoch = None
    best_val_loss = None
    best_val_metrics = None
    best_state = None

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_output_dir / "best_model.pt"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n[{model_name}] Epoch {epoch}/{epochs}")
        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
            print_every=print_every,
            max_batches=max_train_batches,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_metrics["top1"],
            "train_top3": train_metrics["top3"],
            "train_top5": train_metrics["top5"],
        }

        print(
            f"[{model_name}] Train "
            f"loss {train_loss:.4f} | "
            f"top1 {train_metrics['top1']:.2f}% | "
            f"top3 {train_metrics['top3']:.2f}% | "
            f"top5 {train_metrics['top5']:.2f}%"
        )

        if val_loader is not None:
            val_loss, val_metrics = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_amp=use_amp,
                label_smoothing=label_smoothing,
                max_batches=max_eval_batches,
            )
            epoch_record.update(
                {
                    "val_loss": val_loss,
                    "val_top1": val_metrics["top1"],
                    "val_top3": val_metrics["top3"],
                    "val_top5": val_metrics["top5"],
                }
            )

            print(
                f"[{model_name}] Val   "
                f"loss {val_loss:.4f} | "
                f"top1 {val_metrics['top1']:.2f}% | "
                f"top3 {val_metrics['top3']:.2f}% | "
                f"top5 {val_metrics['top5']:.2f}%"
            )

            is_better = best_val_metrics is None or val_metrics["top1"] > best_val_metrics["top1"]
            if is_better:
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_metrics = dict(val_metrics)
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, checkpoint_path)
                print(f"[{model_name}] New best checkpoint saved to {checkpoint_path}")

        history.append(epoch_record)

    elapsed_minutes = (time.time() - start_time) / 60.0

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
        best_val_loss = None
        best_val_metrics = None
        torch.save(model.state_dict(), checkpoint_path)

    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        use_amp=use_amp,
        label_smoothing=label_smoothing,
        max_batches=max_eval_batches,
    )

    history_path = model_output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    result = RunResult(
        model=model_name,
        family=MODEL_PRESETS[model_name]["family"],
        params=params,
        trainable_params=trainable,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_top1=(best_val_metrics["top1"] if best_val_metrics else None),
        best_val_top3=(best_val_metrics["top3"] if best_val_metrics else None),
        best_val_top5=(best_val_metrics["top5"] if best_val_metrics else None),
        test_loss=test_loss,
        test_top1=test_metrics["top1"],
        test_top3=test_metrics["top3"],
        test_top5=test_metrics["top5"],
        train_minutes=elapsed_minutes,
        checkpoint_path=str(checkpoint_path),
        augment=augment,
    )

    return result, history


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_results_table(results: list[RunResult]):
    if not results:
        return "No results."

    rows = []
    for result in results:
        rows.append(
            {
                "model": result.model,
                "params_m": f"{result.params / 1_000_000:.2f}",
                "best_epoch": "-" if result.best_epoch is None else str(result.best_epoch),
                "val_top1": "-" if result.best_val_top1 is None else f"{result.best_val_top1:.2f}",
                "test_top1": "-" if result.test_top1 is None else f"{result.test_top1:.2f}",
                "minutes": f"{result.train_minutes:.2f}",
            }
        )

    headers = list(rows[0].keys())
    widths = {header: max(len(header), max(len(row[header]) for row in rows)) for header in headers}
    lines = []
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "-+-".join("-" * widths[header] for header in headers)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def compare_models(
    model_names: list[str],
    dataset: str = ARENA_DEFAULTS["dataset"],
    data_dir: str = "./data",
    output_dir: str | None = None,
    augment: str = ARENA_DEFAULTS["augment"],
    img_size: int = ARENA_DEFAULTS["img_size"],
    num_classes: int | None = None,
    in_chans: int = 3,
    batch_size: int = ARENA_DEFAULTS["batch_size"],
    eval_batch_size: int = ARENA_DEFAULTS["eval_batch_size"],
    epochs: int = ARENA_DEFAULTS["epochs"],
    lr: float = ARENA_DEFAULTS["lr"],
    weight_decay: float = ARENA_DEFAULTS["weight_decay"],
    val_split: float = ARENA_DEFAULTS["val_split"],
    num_workers: int = 2,
    device: str | None = None,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    warmup_ratio: float = ARENA_DEFAULTS["warmup_ratio"],
    min_lr: float = ARENA_DEFAULTS["min_lr"],
    label_smoothing: float = ARENA_DEFAULTS["label_smoothing"],
    print_every: int = 100,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    seed: int = ARENA_DEFAULTS["seed"],
):
    seed_everything(seed)

    dataset_info = get_dataset_info(dataset)
    resolved_num_classes = num_classes or dataset_info["num_classes"]
    if resolved_num_classes != dataset_info["num_classes"]:
        raise ValueError("num_classes must match the selected built-in arena dataset.")
    if in_chans != 3:
        raise ValueError("Built-in arena datasets emit RGB tensors; use in_chans=3.")

    resolved_models = resolve_model_names(model_names)
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if chosen_device.startswith("cuda") and not torch.cuda.is_available():
        chosen_device = "cpu"

    output_path = (
        Path(output_dir)
        if output_dir
        else REPO_ROOT / "arena_runs" / time.strftime("%Y%m%d-%H%M%S")
    )
    output_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_classification_dataloaders(
        dataset_name=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        img_size=img_size,
        num_workers=num_workers,
        val_split=val_split,
        seed=seed,
        pin_memory=chosen_device.startswith("cuda"),
        augment=augment,
    )
    split_info = {
        "seed": seed,
        "val_split": val_split,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset) if val_loader is not None else 0,
        "test_size": len(test_loader.dataset),
        "augment": augment,
    }

    manifest = {
        "models": resolved_models,
        "dataset": dataset,
        "augment": augment,
        "augment_description": AUGMENT_PRESETS[augment],
        "device": chosen_device,
        "img_size": img_size,
        "num_classes": resolved_num_classes,
        "in_chans": in_chans,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "num_workers": num_workers,
        "autocast_dtype": autocast_dtype,
        "use_amp": use_amp,
        "grad_clip_norm": grad_clip_norm,
        "warmup_ratio": warmup_ratio,
        "min_lr": min_lr,
        "label_smoothing": label_smoothing,
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "seed": seed,
        "split_info": split_info,
    }
    _write_json(output_path / "arena_manifest.json", manifest)

    results = []
    all_histories = {}
    for model_name in resolved_models:
        model, resolved_config = build_model(
            model_name,
            img_size=img_size,
            num_classes=resolved_num_classes,
            in_chans=in_chans,
        )
        _write_json(output_path / model_name / "resolved_model_config.json", resolved_config)
        result, history = train_model_for_arena(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=output_path,
            augment=augment,
            epochs=epochs,
            device=chosen_device,
            lr=lr,
            weight_decay=weight_decay,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr,
            label_smoothing=label_smoothing,
            print_every=print_every,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        )
        results.append(result)
        all_histories[model_name] = history

    serializable_results = [asdict(result) for result in results]
    _write_json(output_path / "arena_results.json", serializable_results)
    _write_csv(output_path / "arena_results.csv", serializable_results)
    _write_json(output_path / "arena_histories.json", all_histories)

    return {
        "output_dir": str(output_path),
        "manifest": manifest,
        "results": results,
        "table": format_results_table(results),
    }
