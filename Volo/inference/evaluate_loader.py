import torch
from typing import Optional, Tuple


@torch.no_grad()
def evaluate_classifier(
    model,
    loader,
    device: str = "cuda",
    amp: bool = True,
    amp_dtype: str = "bf16",
    num_classes: Optional[int] = None,
    topk: Tuple[int, ...] = (1, 3, 5),
    return_confusion: bool = False,):

    """
    Evalua loss + topk en un DataLoader. Asume que el loader retorna (images, targets).
    - return_confusion: si True, retorna matriz de confusion (requiere num_classes).
    """

    import torch.nn.functional as F

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_n = 0
    correct_at_k = {k: 0 for k in topk}

    if return_confusion:
        if num_classes is None:
            raise ValueError("num_classes is required for confusion matrix.")
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    if amp_dtype.lower() == "bf16":
        autocast_dtype = torch.bfloat16
    elif amp_dtype.lower() == "fp16":
        autocast_dtype = torch.float16
    else:
        raise ValueError("amp_dtype must be 'bf16' or 'fp16'")

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=(amp and device.startswith("cuda"))):
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(y.view(1, -1).expand_as(pred.t()))
        for k in topk:
            correct_at_k[k] += int(correct[:k].reshape(-1).float().sum().item())

        if return_confusion:
            y_hat = logits.argmax(dim=1)
            for t, p in zip(y.view(-1), y_hat.view(-1)):
                cm[int(t), int(p)] += 1

    out = {
        "loss": total_loss / max(1, total_n),}

    for k in topk:
        out[f"top{k}"] = 100.0 * correct_at_k[k] / max(1, total_n)

    if return_confusion:
        out["confusion_matrix"] = cm.cpu().numpy()

    return out
