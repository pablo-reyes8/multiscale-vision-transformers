import os, math, random, inspect
from contextlib import contextmanager, nullcontext
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int = 0, deterministic: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


        