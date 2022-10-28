from __future__ import annotations
import os

import torch
import torch.distributed


def init() -> tuple[int, bool, int, torch.device]:
    nranks = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"])
    nranks = max(1, nranks)
    is_distributed = nranks > 1

    num_gpus = torch.cuda.device_count()
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = torch.distributed.get_rank()
    device_id = torch.device(rank % num_gpus)

    if rank == 0:
        print("nranks =", nranks, "\t num_gpus =", torch.cuda.device_count())

    return nranks, is_distributed, rank, device_id


def barrier(rank: int) -> None:
    if rank >= 0:
        torch.distributed.barrier()
