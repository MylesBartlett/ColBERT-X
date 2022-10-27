from __future__ import annotations
import os

import torch
import torch.distributed


def init(rank: int) -> tuple[int, bool]:
    nranks = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"])
    nranks = max(1, nranks)
    is_distributed = nranks > 1

    if rank == 0:
        print("nranks =", nranks, "\t num_gpus =", torch.cuda.device_count())

    if is_distributed:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return nranks, is_distributed


def barrier(rank: int) -> None:
    if rank >= 0:
        torch.distributed.barrier()
