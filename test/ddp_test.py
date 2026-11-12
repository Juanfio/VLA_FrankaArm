import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(f"[START] Rank {rank}/{world_size}", flush=True)

dist.init_process_group(
    backend="gloo",
    init_method="env://",
    timeout=timedelta(minutes=5)
)

print(f"[INITIALIZED] Rank {rank}/{world_size}", flush=True)

dist.barrier()

if rank == 0:
    print("All processes initialized successfully!", flush=True)

dist.destroy_process_group()