"""
Minimal torchrun distributed test — verifies NCCL communication across all GPUs.

Usage (via submit_torchrun_job.sh):
    ./submit_torchrun_job.sh 1 --queue=gpu_l4_parallel --script=test_torchrun.py --venv=~/ray_env
    ./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --script=test_torchrun.py --venv=~/ray_env

Or directly with torchrun:
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
        --master_addr=127.0.0.1 --master_port=29500 test_torchrun.py
"""

import os
import socket
import time
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)

    print(f"[Rank {rank}/{world_size}] host={socket.gethostname()} "
          f"local_rank={local_rank} gpu={gpu_name}")

    # Barrier to sync all ranks before testing
    dist.barrier()

    # Test 1: allreduce — each rank contributes its rank number
    tensor = torch.tensor([float(rank)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    assert tensor.item() == expected, f"allreduce failed: got {tensor.item()}, expected {expected}"
    if rank == 0:
        print(f"[PASS] allreduce: sum of ranks = {tensor.item()} (expected {expected})")

    # Test 2: allgather — collect a unique value from each rank
    local_tensor = torch.tensor([rank * 10.0], device=device)
    gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    gathered_values = [t.item() for t in gathered]
    expected_values = [i * 10.0 for i in range(world_size)]
    assert gathered_values == expected_values, f"allgather failed: {gathered_values}"
    if rank == 0:
        print(f"[PASS] allgather: {gathered_values}")

    # Test 3: bandwidth test — allreduce a larger tensor and measure time
    size_mb = 256
    num_elements = size_mb * 1024 * 1024 // 4  # float32
    big_tensor = torch.randn(num_elements, device=device)
    dist.barrier()

    # Warmup
    for _ in range(3):
        dist.all_reduce(big_tensor)
    torch.cuda.synchronize()

    # Timed runs
    num_iters = 10
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(big_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # allreduce moves 2*(N-1)/N * size bytes (ring algorithm)
    algo_bw = (2.0 * (world_size - 1) / world_size * size_mb * num_iters) / elapsed
    if rank == 0:
        print(f"[PASS] bandwidth: {size_mb}MB allreduce x{num_iters} in {elapsed:.3f}s "
              f"— algo bandwidth: {algo_bw:.1f} MB/s ({algo_bw/1024:.2f} GB/s)")

    dist.barrier()
    if rank == 0:
        print(f"\nAll tests passed — {world_size} GPUs communicating via NCCL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
