#!/usr/bin/env python3
"""
CIFAR-10 distributed training with Ray.
Works on all Janelia GPU queues — IB and Ethernet NCCL.
"""
import argparse
import os
import ray
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

from resnet18_cifar import ResNet18

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus",    type=int, required=True)
    parser.add_argument("--num-nodes",   type=int, required=True)
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume",      type=str, default=None)
    return parser.parse_args()

def get_cifar10_dataloaders(batch_size, world_size, rank):
    """
    Build CIFAR-10 DataLoaders.
    pin_memory=True enables GPUDirect on IB nodes (no-op cost on Ethernet).
    num_workers scales with available CPUs: 8 for H100/H200 (96 CPUs),
    4 for L4 (64 CPUs), safe default 4 otherwise.
    """
    data_path = "/nrs/ml_datasets/cifar10"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True,  download=False, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=False, transform=transform_test)

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    test_sampler  = DistributedSampler(testset,  num_replicas=world_size, rank=rank)

    # Detect CPU count to size workers appropriately
    import multiprocessing
    cpu_count  = multiprocessing.cpu_count()
    # Divide by GPUs per node (8), cap at 8
    num_workers = min(8, max(4, cpu_count // 8))

    # pin_memory required for GPUDirect RDMA on IB nodes
    # persistent_workers avoids respawn cost across epochs
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             sampler=train_sampler, num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)
    testloader  = DataLoader(testset,  batch_size=batch_size,
                             sampler=test_sampler,  num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)

    return trainloader, testloader, train_sampler

def verify_ib_in_use():
    """Verify IB HCAs are active and Ethernet NIC (mlx5_5) is excluded.
    Call once from world_rank=0 at job startup."""
    import subprocess
    ib_devs = ["mlx5_0","mlx5_1","mlx5_2","mlx5_3",
                "mlx5_4","mlx5_6","mlx5_7","mlx5_8"]
    eth_dev = "mlx5_5"
    print("--- IB device check ---")
    for dev in ib_devs:
        try:
            r = subprocess.run(["ibv_devinfo", "-d", dev],
                               capture_output=True, text=True, timeout=5)
            link  = next((l.strip() for l in r.stdout.splitlines()
                          if "link_layer" in l), "unknown")
            state = next((l.strip() for l in r.stdout.splitlines()
                          if "state:" in l), "unknown")
            ok = "InfiniBand" in link
            print(f"  {'OK  ' if ok else 'WARN'} {dev}: {link} | {state}")
        except Exception as e:
            print(f"  ERR  {dev}: {e}")
    nccl_hca = os.environ.get("NCCL_IB_HCA", "")
    if eth_dev in nccl_hca:
        print(f"  ERROR: {eth_dev} (Ethernet) is in NCCL_IB_HCA — remove it!")
    else:
        print(f"  OK   {eth_dev} (Ethernet) correctly excluded from NCCL_IB_HCA")
    print("-----------------------")

def save_checkpoint(model, optimizer, epoch, train_acc, test_acc,
                    config, is_best=False, world_rank=0):
    if world_rank != 0:
        return None
    model_dir = config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    # Always save latest (overwritten each epoch) + best (only when improved)
    suffix = "best" if is_best else "latest"
    path   = os.path.join(model_dir, f"cifar10_resnet18_{suffix}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_accuracy":       train_acc,
        "test_accuracy":        test_acc,
    }, path)
    print(f"Checkpoint saved: {path}  (test acc {test_acc:.2f}%)")
    return path

def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    best_test_acc = ckpt.get("test_accuracy", 0)
    print(f"Resumed from epoch {start_epoch}, best acc {best_test_acc:.2f}%")
    return start_epoch, best_test_acc

def train_func(config):
    import ray.train
    import torch.distributed as dist

    world_rank  = ray.train.get_context().get_world_rank()
    world_size  = ray.train.get_context().get_world_size()

    # 1. Create model on CPU, let prepare_model handle device + DDP
    model = ResNet18(num_classes=10)
    model = ray.train.torch.prepare_model(model)
    device = ray.train.torch.get_device()

    if world_rank == 0:
        print(f"Cluster: {world_size} workers across {config['num_nodes']} nodes")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
        if nccl_ib == "0":
            print("Network: InfiniBand (GPUDirect RDMA)")
            verify_ib_in_use()
            print("Tip: grep 'NET/IB' in .out to confirm NCCL selected IB")
        else:
            print("Network: Ethernet")
        if config.get("save_models"):
            print(f"Checkpoints: {config['model_dir']}")
        if config.get("resume_checkpoint"):
            print(f"Resuming from: {config['resume_checkpoint']}")

    trainloader, testloader, train_sampler = get_cifar10_dataloaders(
        config["batch_size"], world_size, world_rank)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"])

    best_test_acc = 0
    start_epoch   = 0

    if config.get("resume_checkpoint") and world_rank == 0:
        start_epoch, best_test_acc = load_checkpoint(
            config["resume_checkpoint"], model, optimizer, device)

    if world_size > 1:
        se = torch.tensor(start_epoch,   device=device)
        bt = torch.tensor(best_test_acc, device=device)
        dist.broadcast(se, src=0)
        dist.broadcast(bt, src=0)
        start_epoch   = int(se.item())
        best_test_acc = bt.item()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = correct = total = 0

        for batch_idx, (data, targets) in enumerate(trainloader):
            # non_blocking=True overlaps H2D transfer with compute
            # requires pin_memory=True on the DataLoader
            data    = data.to(device,    non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            # AllReduce gradient sync — NCCL routes over IB or Ethernet
            optimizer.step()

            epoch_loss += loss.item()
            total      += targets.size(0)
            correct    += output.argmax(1).eq(targets).sum().item()

            if batch_idx % 50 == 0 and world_rank == 0:
                print(f"  Epoch {epoch} [{batch_idx}/{len(trainloader)}] "
                      f"loss={loss.item():.4f}")

        scheduler.step()
        train_loss = epoch_loss / len(trainloader)
        train_acc  = 100. * correct / total

        model.eval()
        test_loss = test_correct = test_total = 0
        with torch.no_grad():
            for data, targets in testloader:
                data    = data.to(device,    non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                out     = model(data)
                test_loss    += criterion(out, targets).item()
                test_total   += targets.size(0)
                test_correct += out.argmax(1).eq(targets).sum().item()

        test_loss /= len(testloader)
        test_acc   = 100. * test_correct / test_total

        if config.get("save_models"):
            save_checkpoint(model, optimizer, epoch, train_acc, test_acc,
                            config, is_best=False, world_rank=world_rank)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_checkpoint(model, optimizer, epoch, train_acc, test_acc,
                                config, is_best=True, world_rank=world_rank)

        ray.train.report({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0], "best_test_acc": best_test_acc,
        })

        if world_rank == 0:
            print(f"Epoch {epoch}: train={train_acc:.2f}%  test={test_acc:.2f}%")

    if world_rank == 0:
        print(f"\nTraining complete. Best test accuracy: {best_test_acc:.2f}%")

def main():
    args = parse_args()

    dataset_path = "/nrs/ml_datasets/cifar10"
    if not os.path.exists(os.path.join(dataset_path, "cifar-10-batches-py")):
        print(f"ERROR: CIFAR-10 dataset not found at {dataset_path}")
        print("Run the dataset setup step first.")
        return

    submission_dir = os.path.abspath(os.getcwd())
    model_dir      = os.path.join(submission_dir, "models")

    ray.init(address="auto")
    print(f"Ray resources: {ray.available_resources()}")

    # Report active network backend
    nccl_ib  = os.environ.get("NCCL_IB_DISABLE", "1")
    backend  = "InfiniBand" if nccl_ib == "0" else "Ethernet"
    print(f"NCCL backend: {backend}")

    scaling_config = ScalingConfig(
        num_workers=args.num_gpus,
        use_gpu=True,
        resources_per_worker={"CPU": 7, "GPU": 1},
    )
    run_config = RunConfig(
        name="cifar10_resnet18",
        storage_path="/scratch/{}/ray_results".format(
            os.getenv("USER", "unknown")),
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr":                args.lr,
            "batch_size":        args.batch_size,
            "epochs":            args.epochs,
            "num_nodes":         args.num_nodes,
            "save_models":       args.save_models,
            "resume_checkpoint": args.resume,
            "model_dir":         model_dir,
        },
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print(f"Starting training: {args.num_gpus} GPUs across {args.num_nodes} nodes")
    result = trainer.fit()

    if result and result.metrics:
        print(f"Final test accuracy: {result.metrics['test_acc']:.2f}%")
        print(f"Best test accuracy:  {result.metrics['best_test_acc']:.2f}%")

    ray.shutdown()

if __name__ == "__main__":
    main()
