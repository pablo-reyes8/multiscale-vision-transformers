import torch 
import torch.nn as nn

def _fmt_out(output):
    if isinstance(output, (tuple, list)):
        shapes = []
        for o in output:
            if hasattr(o, "shape"):
                shapes.append(tuple(o.shape))
            else:
                shapes.append(type(o).__name__)
        return shapes
    if hasattr(output, "shape"):
        return tuple(output.shape)
    return type(output).__name__


def attach_shape_hooks_volo(model: nn.Module, verbose: bool = True):
    hooks = []

    def add_hook(mod: nn.Module, name: str):
        if mod is None:
            return
        def hook(_m, _inp, out):
            print(f"{name:35s} -> {_fmt_out(out)}")
        hooks.append(mod.register_forward_hook(hook))

    # Top-level components
    add_hook(getattr(model, "patch_embed", None), "patch_embed")
    add_hook(getattr(model, "local_stage", None), "local_stage (outlooker)")
    add_hook(getattr(model, "pyramid", None), "pyramid (top)")
    add_hook(getattr(model, "norm", None), "norm")
    add_hook(getattr(model, "head", None), "head")

    # Global blocks (flat)
    if hasattr(model, "global_blocks"):
        for i, blk in enumerate(model.global_blocks):
            add_hook(blk, f"global_block[{i}]")

    # Pyramid internals (hierarchical)
    pyr = getattr(model, "pyramid", None)
    if pyr is not None:
        if hasattr(pyr, "levels"):
            for i, lvl in enumerate(pyr.levels):
                # lvl es nn.ModuleDict: NO tiene .get
                loc = lvl["local"] if "local" in lvl else None
                glob = lvl["global"] if "global" in lvl else None
                add_hook(loc,  f"pyr.level[{i}].local")
                add_hook(glob, f"pyr.level[{i}].global")

        if hasattr(pyr, "downsamples"):
            for i, ds in enumerate(pyr.downsamples):
                add_hook(ds, f"pyr.down[{i}]")

    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

@torch.no_grad()
def debug_forward_shapes(model: nn.Module, img_size: int, device: str = "cpu", batch_size: int = 2):
    model = model.to(device).eval()
    hooks = attach_shape_hooks_volo(model)

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    print(f"\n=== Forward debug | img_size={img_size} | model={model.__class__.__name__} ===")
    y = model(x)
    print(f"{'OUTPUT logits':35s} -> {tuple(y.shape)}")

    remove_hooks(hooks)