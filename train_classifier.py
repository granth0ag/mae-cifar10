import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model import *
from utils import setup_seed

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier.pt')
    args = parser.parse_args()

    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup_seed(args.seed + global_rank)
    torch.backends.cudnn.benchmark = True

    # Compute batch split per GPU
    assert args.batch_size % world_size == 0, "batch_size must be divisible by world_size"
    per_gpu_batch_size = args.batch_size // world_size  # 128 // 2 = 64 per T4

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=(global_rank == 0), transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=(global_rank == 0), transform=transform)
    dist.barrier()  # Block rank 1 until rank 0 finishes checking/downloading dataset

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=train_sampler, 
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=val_sampler, 
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True
    )

    # Initialize model
    base_mae = MAE_ViT()
    if args.pretrained_model_path is not None:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
        base_mae.load_state_dict(state_dict)
        writer_tag = 'pretrain-cls'
    else:
        writer_tag = 'scratch-cls'

    model = ViT_Classifier(base_mae.encoder, num_classes=10).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    writer = SummaryWriter(os.path.join('logs', 'cifar10', writer_tag)) if global_rank == 0 else None

    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.base_learning_rate * args.batch_size / 256, 
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay
    )
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epoch + 1e-8), 
        0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    best_val_acc = 0.0
    optim.zero_grad(set_to_none=True)

    for e in range(args.total_epoch):
        train_sampler.set_epoch(e)
        model.train()
        
        running_train_loss = torch.tensor(0.0, device=local_rank)
        running_train_correct = torch.tensor(0.0, device=local_rank)
        train_samples = 0

        pbar = tqdm(train_dataloader, desc=f"epoch {e}") if global_rank == 0 else train_dataloader

        for img, label in pbar:
            img = img.to(local_rank, non_blocking=True)
            label = label.to(local_rank, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(img)
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            # Track stats locally on GPU without .item() sync stalls
            bs = img.size(0)
            train_samples += bs
            running_train_loss += loss.detach() * bs
            running_train_correct += (logits.argmax(dim=-1) == label).float().sum()

        lr_scheduler.step()

        # Reduce training metrics across all ranks
        dist.all_reduce(running_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_train_correct, op=dist.ReduceOp.SUM)
        total_train_samples = torch.tensor(train_samples, device=local_rank)
        dist.all_reduce(total_train_samples, op=dist.ReduceOp.SUM)

        avg_train_loss = (running_train_loss / total_train_samples).item()
        avg_train_acc = (running_train_correct / total_train_samples).item()

        # Validation pass
        model.eval()
        running_val_loss = torch.tensor(0.0, device=local_rank)
        running_val_correct = torch.tensor(0.0, device=local_rank)
        val_samples = 0

        with torch.no_grad():
            for img, label in val_dataloader:
                img = img.to(local_rank, non_blocking=True)
                label = label.to(local_rank, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(img)
                    loss = loss_fn(logits, label)

                bs = img.size(0)
                val_samples += bs
                running_val_loss += loss.detach() * bs
                running_val_correct += (logits.argmax(dim=-1) == label).float().sum()

        # Reduce validation metrics across all ranks
        dist.all_reduce(running_val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_val_correct, op=dist.ReduceOp.SUM)
        total_val_samples = torch.tensor(val_samples, device=local_rank)
        dist.all_reduce(total_val_samples, op=dist.ReduceOp.SUM)

        avg_val_loss = (running_val_loss / total_val_samples).item()
        avg_val_acc = (running_val_correct / total_val_samples).item()

        # Logging and checkpointing exclusively on master rank
        if global_rank == 0:
            print(f"Epoch {e}: train_loss = {avg_train_loss:.4f}, train_acc = {avg_train_acc:.4f} | val_loss = {avg_val_loss:.4f}, val_acc = {avg_val_acc:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                print(f"-> saving checkpoint with val_acc = {best_val_acc:.4f} at epoch {e}")
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'val_acc': best_val_acc,
                }, args.output_model_path)

            writer.add_scalars('cls/loss', {'train': avg_train_loss, 'val': avg_val_loss}, global_step=e)
            writer.add_scalars('cls/acc', {'train': avg_train_acc, 'val': avg_val_acc}, global_step=e)

    if global_rank == 0:
        writer.close()
    destroy_process_group()

if __name__ == '__main__':
    main()