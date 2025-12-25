def attach_shape_hooks(model):
    hooks = []

    def mk_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                out_shape = [tuple(o.shape) for o in output if hasattr(o, "shape")]
            else:
                out_shape = tuple(output.shape) if hasattr(output, "shape") else type(output)
            print(f"{name:20s} -> {out_shape}")
        return hook

    hooks.append(model.stem.register_forward_hook(mk_hook("stem")))

    for i, st in enumerate(model.stages):
        hooks.append(st.register_forward_hook(mk_hook(f"stage{i}")))
        if i < len(model.downsamples):
            hooks.append(model.downsamples[i].register_forward_hook(mk_hook(f"down{i}")))

    hooks.append(model.pool.register_forward_hook(mk_hook("pool")))
    hooks.append(model.head.register_forward_hook(mk_hook("head")))
    return hooks