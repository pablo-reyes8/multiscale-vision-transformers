import torch

def accuracy_topk(logits: torch.Tensor,
                  targets: torch.Tensor,
                  ks=(1, 3, 5)):
    """
    Calcula accuracy@k para una lista de k.

    Args:
        logits: [B, num_classes]
        targets: [B] con labels enteros
        ks: iterable de ints, por ejemplo (1,3,5)

    Returns:
        dict {k: acc_k} con accuracys en porcentaje (0-100).
    """
    with torch.no_grad():
        max_k = max(ks)
        B = targets.size(0)

        _, pred = torch.topk(logits, k=max_k, dim=1)
        targets_exp = targets.view(-1, 1).expand_as(pred)

        correct = pred.eq(targets_exp)  #
        accs = {}

        for k in ks:
            correct_k = correct[:, :k].any(dim=1).float().sum().item()
            accs[k] = 100.0 * correct_k / B

    return accs