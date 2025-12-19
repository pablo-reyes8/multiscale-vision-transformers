import torch
import torch.nn.functional as F

def pad_to_window_size(x: torch.Tensor, window_size: int, pad_value: float = 0.0):
    """
    Asegura que H y W sean múltiplos de window_size mediante padding (bottom/right).

    Args:
        x: [B, H, W, C]
        window_size: int
        pad_value: valor de relleno

    Returns:
        x_pad: [B, Hp, Wp, C]
        pad_hw: (pad_h, pad_w)
        orig_hw: (H, W)
    """
    assert x.dim() == 4, "x debe ser [B, H, W, C]"
    B, H, W, C = x.shape

    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0), (H, W)

    # F.pad para tensores 4D en este orden: (..., C) => (pad_C_left, pad_C_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right)
    x_pad = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), value=pad_value)
    return x_pad, (pad_h, pad_w), (H, W)

def unpad_from_window_size(x_pad: torch.Tensor, orig_hw: tuple[int, int]):
    """
    Revierte el padding, recortando a (H, W) original.

    Args:
        x_pad: [B, Hp, Wp, C]
        orig_hw: (H, W)

    Returns:
        x: [B, H, W, C]
    """
    H, W = orig_hw
    return x_pad[:, :H, :W, :]

def window_partition(x: torch.Tensor, window_size: int):
    """
    Divide [B, H, W, C] en ventanas no solapadas.

    Args:
        x: [B, H, W, C] con H, W múltiplos de window_size
        window_size: int

    Returns:
        windows: [num_windows * B, window_size, window_size, C]
    """
    assert x.dim() == 4, "x debe ser [B, H, W, C]"

    B, H, W, C = x.shape

    assert H % window_size == 0 and W % window_size == 0, \
        f"H,W deben ser múltiplos de window_size. Got {(H, W)} vs {window_size}"

    x = x.view(B,
        H // window_size, window_size,
        W // window_size, window_size, C)

    # [B, nH, ws, nW, ws, C] -> [B, nH, nW, ws, ws, C] -> [B*nH*nW, ws, ws, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows



def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int | None = None):
    """
    Reconstruye [B, H, W, C] a partir de ventanas.

    Args:
        windows: [num_windows * B, window_size, window_size, C]
        window_size: int
        H, W: alto y ancho del mapa (padded) que queremos reconstruir
        B: batch size (si None, se infiere)

    Returns:
        x: [B, H, W, C]
    """
    assert windows.dim() == 4, "windows debe ser [B*nW, ws, ws, C]"
    nBW, ws1, ws2, C = windows.shape
    assert ws1 == window_size and ws2 == window_size

    nH = H // window_size
    nW = W // window_size

    if B is None:
        assert nBW % (nH * nW) == 0, "No puedo inferir B: shapes incompatibles."
        B = nBW // (nH * nW)

    x = windows.view(B, nH, nW, window_size, window_size, C)
    # [B, nH, nW, ws, ws, C] -> [B, nH, ws, nW, ws, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x



def cyclic_shift(x: torch.Tensor, shift_size: int | tuple[int, int]):
    """
    Swin usa un shift circular (torch.roll) en H y W.

    Args:
        x: [B, H, W, C]
        shift_size: int o (shift_h, shift_w)
            - shift positivo desplaza hacia abajo/derecha
            - shift negativo hacia arriba/izquierda (lo típico en SW-MSA es negativo)

    Returns:
        x_shifted: [B, H, W, C]
    """
    if isinstance(shift_size, int):
        shift_h, shift_w = shift_size, shift_size
    else:
        shift_h, shift_w = shift_size
    return torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))


@torch.no_grad()
def build_shifted_window_attention_mask(
    H: int,
    W: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32):
    """
    Crea la máscara que evita que, tras el shift, una ventana atienda a tokens que
    originalmente venían de regiones distintas (bordes "cortados" por el shift).

    Convención:
      - máscara devuelve shape [num_windows, ws*ws, ws*ws]
      - valores: 0 para permitido, -inf (o gran negativo) para bloqueado

    Nota: H y W deben ser múltiplos de window_size (usa pad_to_window_size antes).

    Args:
        H, W: dimensiones (padded)
        window_size: ws
        shift_size: ss (0 < ss < ws)
        device: device
        dtype: dtype (float)

    Returns:
        attn_mask: [num_windows, ws*ws, ws*ws]
    """
    assert 0 < shift_size < window_size, "shift_size debe estar en (0, window_size)."

    img_mask = torch.zeros((1, H, W, 1), device=device, dtype=torch.int64)

    # Dividimos H y W en 3 bandas: [0:-ws], [-ws:-ss], [-ss:]
    h_slices = (slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),)

    w_slices = (slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),)

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # Particionamos esa máscara en ventanas
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)

    # Diferencias: si id distinto => bloquear atención
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.ne(0)  # bool

    # Convertimos a float con -inf para posiciones bloqueadas
    neg_inf = torch.finfo(dtype).min
    attn_mask = attn_mask.to(dtype) * neg_inf
    return attn_mask


def prepare_windows(
    x: torch.Tensor,
    window_size: int,
    shift_size: int = 0,
    pad_value: float = 0.0):
    """
    Prepara todo lo necesario "antes de la atención":
    - padding a múltiplos de window_size
    - shift opcional (SW-MSA)
    - window_partition
    - attention mask (si shift_size > 0)

    Args:
        x: [B, H, W, C]
        window_size: ws
        shift_size: ss (0 para W-MSA, >0 para SW-MSA)
        pad_value: padding

    Returns:
        windows_flat: [B*num_windows, ws*ws, C]  (listo para entrar a atención)
        meta: dict con info para reconstruir (reverse)
              incluye (Hp, Wp), (H, W), pad_hw, shift_size, attn_mask
    """
    assert x.dim() == 4, "x debe ser [B, H, W, C]"
    B, H, W, C = x.shape
    device = x.device

    x_pad, pad_hw, orig_hw = pad_to_window_size(x, window_size, pad_value=pad_value)
    _, Hp, Wp, _ = x_pad.shape

    attn_mask = None
    if shift_size > 0:
        x_pad = cyclic_shift(x_pad, (-shift_size, -shift_size))  # shift negativo (convención Swin)

        attn_mask = build_shifted_window_attention_mask(
            H=Hp, W=Wp,
            window_size=window_size,
            shift_size=shift_size,
            device=device,
            dtype=x.dtype,)

    # [B*nW, ws, ws, C]
    windows = window_partition(x_pad, window_size)
    # [B*nW, ws*ws, C]  (forma típica para pasar a atención)
    windows_flat = windows.view(-1, window_size * window_size, C)

    meta = {
        "B": B,
        "orig_hw": orig_hw,   # (H, W) original
        "pad_hw": pad_hw,     # (pad_h, pad_w)
        "HpWp": (Hp, Wp),     # (Hp, Wp) padded
        "window_size": window_size,
        "shift_size": shift_size,
        "attn_mask": attn_mask}

    return windows_flat, meta

def restore_from_windows(
    windows_flat: torch.Tensor,
    meta: dict,
    C: int):

    """
    Revierte la preparación:
    - windows_flat -> windows -> window_reverse -> unshift (si aplica) -> unpad

    Args:
        windows_flat: [B*nW, ws*ws, C]
        meta: dict devuelto por prepare_windows
        C: canales

    Returns:
        x: [B, H, W, C] (H,W original)
    """
    B = meta["B"]
    (Hp, Wp) = meta["HpWp"]
    ws = meta["window_size"]
    ss = meta["shift_size"]
    orig_hw = meta["orig_hw"]

    windows = windows_flat.view(-1, ws, ws, C)
    x_pad = window_reverse(windows, ws, Hp, Wp, B=B)

    if ss > 0:
        x_pad = cyclic_shift(x_pad, (ss, ss))  # deshacer shift

    x = unpad_from_window_size(x_pad, orig_hw)
    return x
