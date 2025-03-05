import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def run(rank, world_size):
    print(f"Running on GPU {rank}")
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29500", rank=rank, world_size=world_size)
    tensor = torch.ones(1).cuda(rank)
    dist.all_reduce(tensor)
    print(f"Rank {rank} has tensor {tensor}")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
