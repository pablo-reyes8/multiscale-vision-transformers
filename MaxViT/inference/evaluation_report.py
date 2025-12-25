from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Any
import math
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

def _as_np(x):
    # history puede venir como list[float] o np.array
    return np.asarray(x, dtype=float)

def history_len(history: Dict[str, Any]) -> int:
    # toma la longitud máxima entre series
    return int(max(len(v) for v in history.values()))

def best_epoch(history: Dict[str, Any], metric: str = "val_top1", mode: str = "max") -> int:
    """
    Retorna el índice de epoch (0-based) que optimiza `metric`.
    mode: "max" (accuracy) o "min" (loss)
    """
    if metric not in history:
        raise KeyError(f"Metric '{metric}' not found. Available: {list(history.keys())}")
    arr = _as_np(history[metric])
    if mode == "max":
        return int(np.nanargmax(arr))
    elif mode == "min":
        return int(np.nanargmin(arr))
    else:
        raise ValueError("mode must be 'max' or 'min'")

def final_metrics(history: Dict[str, Any]) -> Dict[str, float]:
    """Métricas del último epoch."""
    out = {}
    for k, v in history.items():
        a = _as_np(v)
        if len(a) == 0:
            continue
        out[k] = float(a[-1])
    return out

def epoch_metrics(history: Dict[str, Any], epoch: int) -> Dict[str, float]:
    """Métricas de un epoch específico (0-based)."""
    out = {}
    for k, v in history.items():
        a = _as_np(v)
        if epoch < 0 or epoch >= len(a):
            continue
        out[k] = float(a[epoch])
    return out

def generalization_gaps(history: Dict[str, Any], epoch: Optional[int] = None) -> Dict[str, float]:
    """
    Computa gaps train - val para loss y topk si existen.
    Para loss: gap_loss = val_loss - train_loss (positivo suele ser peor generalización)
    Para acc:  gap_top1 = train_top1 - val_top1 (positivo = train mejor que val)
    """
    if epoch is None:
        epoch = history_len(history) - 1

    m = epoch_metrics(history, epoch)
    gaps = {}

    if "train_loss" in m and "val_loss" in m:
        gaps["gap_loss(val-train)"] = m["val_loss"] - m["train_loss"]

    for k in ["top1", "top3", "top5"]:
        tk, vk = f"train_{k}", f"val_{k}"
        if tk in m and vk in m:
            gaps[f"gap_{k}(train-val)"] = m[tk] - m[vk]

    return gaps

def top_epochs_table(
    history: Dict[str, Any],
    sort_by: str = "val_top1",
    mode: str = "max",
    k: int = 10):
    
    """
    Tabla con los mejores k epochs según `sort_by`.
    Si pandas no está disponible retorna lista de dicts.
    """
    n = history_len(history)
    cols = [c for c in ["train_loss","val_loss","train_top1","val_top1","train_top3","val_top3","train_top5","val_top5"] if c in history]
    rows = []
    for e in range(n):
        row = {"epoch": e + 1}  # 1-based para reporte
        for c in cols:
            a = _as_np(history[c])
            if e < len(a):
                row[c] = float(a[e])
        # gaps por epoch
        g = generalization_gaps(history, epoch=e)
        row.update(g)
        rows.append(row)

    if sort_by not in rows[0]:
        raise KeyError(f"sort_by '{sort_by}' not in table columns. Example keys: {list(rows[0].keys())}")

    rows_sorted = sorted(rows, key=lambda r: (r.get(sort_by, np.nan)), reverse=(mode=="max"))
    best = rows_sorted[:k]

    if pd is None:
        return best
    return pd.DataFrame(best)

def training_report(
    history: Dict[str, Any],
    primary_metric: str = "val_top1",
    primary_mode: str = "max",
    secondary_metric: str = "val_loss",
    secondary_mode: str = "min",
) -> Dict[str, Any]:
    """
    Reporte compacto para imprimir o loggear.
    - Mejor epoch por val_top1 (o el que elijas)
    - Mejor epoch por val_loss (o el que elijas)
    - Métricas finales
    - Gaps finales
    """
    rep = {}
    e_best_primary = best_epoch(history, primary_metric, primary_mode)
    e_best_secondary = best_epoch(history, secondary_metric, secondary_mode)

    rep["best_primary"] = {
        "metric": primary_metric,
        "epoch": e_best_primary + 1,
        "metrics": epoch_metrics(history, e_best_primary),
        "gaps": generalization_gaps(history, e_best_primary),
    }
    rep["best_secondary"] = {
        "metric": secondary_metric,
        "epoch": e_best_secondary + 1,
        "metrics": epoch_metrics(history, e_best_secondary),
        "gaps": generalization_gaps(history, e_best_secondary),}

    rep["final"] = {
        "epoch": history_len(history),
        "metrics": final_metrics(history),
        "gaps": generalization_gaps(history, None),}

    return rep

def pretty_print_report(report: Dict[str, Any]) -> None:
    def _fmt(d: Dict[str, float], keys: Optional[List[str]] = None) -> str:
        if keys is None:
            keys = list(d.keys())
        parts = []
        for k in keys:
            if k in d:
                v = d[k]
                if "loss" in k:
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v:.2f}")
        return ", ".join(parts)

    for block in ["best_primary", "best_secondary", "final"]:
        b = report[block]
        print(f"\n[{block}] epoch={b['epoch']}" if block == "final" else f"\n[{block}] ({b['metric']}) epoch={b['epoch']}")
        print("  metrics:", _fmt(b["metrics"], ["train_loss","val_loss","train_top1","val_top1","train_top3","val_top3","train_top5","val_top5"]))
        print("  gaps:   ", _fmt(b["gaps"]))