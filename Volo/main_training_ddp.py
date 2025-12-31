import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.VOLO import * 
from data.load_data_ddp import * 
from training.Train_VOLO import *


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    train_ds, val_ds, test_ds = get_cifar100_datasets(
        data_dir="./data/cifar100",
        val_split=0.1,
        img_size=32,
        ddp_safe_download=True,)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)

    train_loader = DataLoader(
            train_ds,
            batch_size=256,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,)

    val_loader = None
    
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=256,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,)

    model = VOLOClassifier(
        num_classes=100,
        img_size=32,
        patch_size=4,
        hierarchical=False,
        embed_dim=320,
        outlooker_depth=5,
        outlooker_heads=10,
        transformer_depth=10,
        transformer_heads=10,
        kernel_size=3,
        mlp_ratio=4.0,
        dropout=0.12,
        attn_dropout=0.05,
        drop_path_rate=0.20,
        pooling="cls",
        cls_attn_depth=2,
        use_pos_embed=True,
        use_cls_pos=True,).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank ,find_unused_parameters=True)

    history, best = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=130,
        device=str(device),
        lr=5e-4,
        weight_decay=0.05,
        use_amp=True,
        autocast_dtype="fp16",
        print_every=25,
        num_classes=100,
        save_path="best_model.pt",
        last_path="last_model.pt",)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()