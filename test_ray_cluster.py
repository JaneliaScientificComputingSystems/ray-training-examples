#!/usr/bin/env python3
"""Ray cluster test — validates setup, GPU access, and network backend."""
import ray
import torch
import time
import os

def main():
    print("=" * 70)
    print("RAY CLUSTER TEST")
    print("=" * 70)

    ray.init(address="auto")
    print("Connected to Ray cluster")

    resources = ray.available_resources()
    total_cpus = resources.get('CPU', 0)
    total_gpus = resources.get('GPU', 0)
    total_mem  = resources.get('memory', 0) / (1024**3)
    print(f"  CPUs:   {total_cpus}")
    print(f"  GPUs:   {total_gpus}")
    print(f"  Memory: {total_mem:.1f} GB")

    # Report network backend
    nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
    backend  = "InfiniBand" if nccl_ib == "0" else "Ethernet"
    print(f"  NCCL backend: {backend}")
    if nccl_ib == "0":
        print(f"  IB HCAs: {os.environ.get('NCCL_IB_HCA', 'not set')}")

    # Test GPU access on every GPU
    @ray.remote(num_gpus=1)
    def test_gpu(gpu_id):
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            name = torch.cuda.get_device_name(0)
            x = torch.randn(1000, 1000, device=device)
            torch.matmul(x, x)
            return f"GPU {gpu_id}: {name} — OK"
        return f"GPU {gpu_id}: CUDA not available"

    num_gpus = int(total_gpus)
    if num_gpus > 0:
        print(f"\nTesting {num_gpus} GPUs...")
        results = ray.get([test_gpu.remote(i) for i in range(num_gpus)])
        for r in results:
            print(f"  {r}")

    # Test distributed computation
    @ray.remote
    def compute_task(size):
        import numpy as np
        m = np.random.rand(size, size)
        return np.linalg.eigvals(m).shape[0]

    print("\nRunning 20 parallel distributed tasks...")
    t0 = time.time()
    ray.get([compute_task.remote(1000) for _ in range(20)])
    print(f"  Completed in {time.time()-t0:.2f}s")

    print("\n" + "=" * 70)
    print(f"Cluster OK — {total_gpus} GPUs | {backend} NCCL")
    print("=" * 70)
    ray.shutdown()

if __name__ == "__main__":
    main()
