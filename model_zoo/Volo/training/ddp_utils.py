import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_on() else 0

def is_main_process() -> bool:
    return (not ddp_is_on()) or ddp_rank() == 0

def ddp_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM in-place and return tensor."""
    if ddp_is_on():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def ddp_broadcast_bool(flag: bool, device: torch.device | str) -> bool:
    """Broadcast a stop flag from rank0 to all ranks."""
    t = torch.tensor([1 if flag else 0], device=device)
    if ddp_is_on():
        dist.broadcast(t, src=0)
    return bool(t.item())