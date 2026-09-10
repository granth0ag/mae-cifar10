import os
import argparse
import math

import torch
import torchvision
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import MAE_ViT, rearrange
from utils import setup_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')

    args = parser.parse_args()

    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    if rank == 0:
        torchvision.datasets.CIFAR10(
            'data',
            train=True,
            download=True
        )
        torchvision.datasets.CIFAR10(
            'data',
            train=False,
            download=True
        )

    dist.barrier()
    

    setup_seed(args.seed + rank)

    # batch_size is the global batch size
    assert args.batch_size % world_size == 0

    per_device_batch_size = args.batch_size // world_size

    assert per_device_batch_size % args.max_device_batch_size == 0

    load_batch_size = min(
        args.max_device_batch_size,
        per_device_batch_size
    )

    steps_per_update = per_device_batch_size // load_batch_size

    if rank == 0:
        print(f'GPUs: {world_size}')
        print(f'Global batch size: {args.batch_size}')
        print(f'Per-GPU batch size: {per_device_batch_size}')
        print(f'Batch per forward pass: {load_batch_size}')
        print(f'Gradient accumulation steps: {steps_per_update}')

    transform = Compose([
        ToTensor(),
        Normalize(0.5, 0.5)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        'data',
        train=True,
        download=False,
        transform=transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        'data',
        train=False,
        download=False,
        transform=transform
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=load_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    writer = None
    if rank == 0:
        writer = SummaryWriter(
            os.path.join('logs', 'cifar10', 'mae-pretrain')
        )

    model = MAE_ViT(
        mask_ratio=args.mask_ratio
    ).to(device)

    model = DDP(
        model,
        device_ids=[local_rank]
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_learning_rate * args.batch_size / 256,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epoch + 1e-8),
        0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1)
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lr_func
    )

    scaler = torch.amp.GradScaler('cuda')

    optim.zero_grad()
    step_count = 0

    for e in range(args.total_epoch):
        model.train()
        train_sampler.set_epoch(e)

        losses = []

        if rank == 0:
            dataloader = tqdm(
                train_dataloader,
                desc=f'Epoch {e}'
            )
        else:
            dataloader = train_dataloader

        for img, _ in dataloader:
            step_count += 1

            img = img.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast('cuda'):
                predicted_img, mask = model(img)
                loss = (
                    torch.mean(
                        (predicted_img - img) ** 2 * mask
                    ) / args.mask_ratio
                )

            scaler.scale(
                loss / steps_per_update
            ).backward()

            if step_count % steps_per_update == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            losses.append(loss.item())

        lr_scheduler.step()

        avg_loss = sum(losses) / len(losses)

        if rank == 0:
            current_lr = optim.param_groups[0]['lr']

            writer.add_scalar(
                'train/loss',
                avg_loss,
                global_step=e
            )

            writer.add_scalar(
                'train/lr',
                current_lr,
                global_step=e
            )

            print(
                f'Epoch {e}: '
                f'train_loss = {avg_loss:.5f}, '
                f'lr = {current_lr:.8f}'
            )

        if rank == 0 and (
            e % 10 == 0 or e == args.total_epoch - 1
        ):
            model.eval()

            with torch.no_grad():
                val_img = torch.stack([
                    val_dataset[i][0]
                    for i in range(16)
                ]).to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast('cuda'):
                    predicted_val_img, mask = model(val_img)

                predicted_val_img = (
                    predicted_val_img * mask
                    + val_img * (1 - mask)
                )

                vis = torch.cat([
                    val_img * (1 - mask),
                    predicted_val_img,
                    val_img
                ], dim=0)

                vis = rearrange(
                    vis,
                    '(v h1 w1) c h w -> c (h1 h) (w1 v w)',
                    w1=2,
                    v=3
                )

                writer.add_image(
                    'val/reconstructions',
                    (vis + 1) / 2,
                    global_step=e
                )

        if rank == 0 and (
            e % 10 == 0 or e == args.total_epoch - 1
        ):
            torch.save({
                'epoch': e,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }, args.model_path)

    if writer is not None:
        writer.close()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()