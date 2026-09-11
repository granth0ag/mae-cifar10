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

from model import *
from utils import setup_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument(
        '--output_model_path',
        type=str,
        default='vit-t-classifier-from_scratch.pt'
    )
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    

    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    setup_seed(args.seed + rank)

    batch_size = args.batch_size

    assert batch_size % world_size == 0

    per_device_batch_size = batch_size // world_size

    load_batch_size = min(
        args.max_device_batch_size,
        per_device_batch_size
    )

    assert per_device_batch_size % load_batch_size == 0

    steps_per_update = per_device_batch_size // load_batch_size

    if rank == 0:
        print(f'GPUs: {world_size}')
        print(f'Global batch size: {batch_size}')
        print(f'Per-GPU batch size: {per_device_batch_size}')
        print(f'Batch per forward pass: {load_batch_size}')
        print(f'Gradient accumulation steps: {steps_per_update}')

    if rank == 0:
        torchvision.datasets.CIFAR10(
            'data',
            train=True,
            download=False
        )
        torchvision.datasets.CIFAR10(
            'data',
            train=False,
            download=False
        )

    dist.barrier()

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

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=load_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    writer = None

    if rank == 0:
        if args.pretrained_model_path is not None:
            log_dir = os.path.join(
                'logs',
                'cifar10',
                'pretrain-cls'
            )
        else:
            log_dir = os.path.join(
                'logs',
                'cifar10',
                'scratch-cls'
            )

        writer = SummaryWriter(log_dir)

    if args.pretrained_model_path is not None:

        checkpoint = torch.load(
            args.pretrained_model_path,
            map_location='cpu'
        )

        state_dict = (
            checkpoint.get('model_state_dict', checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint.state_dict()
        )

        base_mae = MAE_ViT()
        base_mae.load_state_dict(state_dict)

        model = ViT_Classifier(
            base_mae.encoder,
            num_classes=10
        ).to(device)

    else:

        base_mae = MAE_ViT()

        model = ViT_Classifier(
            base_mae.encoder,
            num_classes=10
        ).to(device)

    model = DDP(
        model,
        device_ids=[local_rank]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    acc_fn = lambda logit, label: (
        logit.argmax(dim=-1) == label
    ).float().mean()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_learning_rate * args.batch_size / 256,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epoch + 1e-8),
        0.5 * (
            math.cos(
                epoch / args.total_epoch * math.pi
            ) + 1
        )
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lr_func
    )

    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    step_count = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location='cpu'
        )

        model.module.load_state_dict(
            checkpoint['model_state_dict']
        )

        optim.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        lr_scheduler.load_state_dict(
            checkpoint['scheduler_state_dict']
        )

        scaler.load_state_dict(
            checkpoint['scaler_state_dict']
        )

        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']

        if rank == 0:
            print(f'Resuming from epoch {start_epoch}')

    optim.zero_grad()

    for e in range(start_epoch,args.total_epoch):

        model.train()
        train_sampler.set_epoch(e)

        losses = []
        acces = []

        if rank == 0:
            dataloader = tqdm(
                train_dataloader,
                desc=f'epoch {e}'
            )
        else:
            dataloader = train_dataloader

        for img, label in dataloader:

            step_count += 1

            img = img.to(
                device,
                non_blocking=True
            )

            label = label.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast('cuda'):
                logits = model(img)
                loss = loss_fn(logits, label)

            acc = acc_fn(logits, label)

            scaler.scale(
                loss / steps_per_update
            ).backward()

            if step_count % steps_per_update == 0:

                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            losses.append(loss.item())
            acces.append(acc.item())

        lr_scheduler.step()

        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)

        if rank == 0:

            current_lr = optim.param_groups[0]['lr']

            print(
                f'Epoch {e}: '
                f'train_loss = {avg_train_loss:.4f}, '
                f'train_acc = {avg_train_acc:.4f}, '
                f'lr = {current_lr:.8f}'
            )

            writer.add_scalars(
                'cls/loss',
                {'train': avg_train_loss},
                global_step=e
            )

            writer.add_scalars(
                'cls/acc',
                {'train': avg_train_acc},
                global_step=e
            )

            model.module.eval()

            val_losses = []
            val_acces = []

            with torch.no_grad():

                for img, label in val_dataloader:

                    img = img.to(
                        device,
                        non_blocking=True
                    )

                    label = label.to(
                        device,
                        non_blocking=True
                    )

                    with torch.amp.autocast('cuda'):
                        logits = model.module(img)
                        loss = loss_fn(logits, label)

                    acc = acc_fn(logits, label)

                    val_losses.append(loss.item())
                    val_acces.append(acc.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = sum(val_acces) / len(val_acces)

            print(
                f'Epoch {e}: '
                f'val_loss = {avg_val_loss:.4f}, '
                f'val_acc = {avg_val_acc:.4f}'
            )

            writer.add_scalars(
                'cls/loss',
                {'val': avg_val_loss},
                global_step=e
            )

            writer.add_scalars(
                'cls/acc',
                {'val': avg_val_acc},
                global_step=e
            )

            if avg_val_acc > best_val_acc:

                best_val_acc = avg_val_acc

                print(
                    f'-> saving checkpoint with '
                    f'val_acc = {best_val_acc:.4f} '
                    f'at epoch {e}'
                )

                torch.save(
                    {
                        'epoch': e,
                        'model_state_dict':
                            model.module.state_dict(),
                        'optimizer_state_dict':
                            optim.state_dict(),
                        'scheduler_state_dict':
                            lr_scheduler.state_dict(),
                        'scaler_state_dict':
                            scaler.state_dict(),
                        'val_acc': best_val_acc,
                    },
                    args.output_model_path
                )

    if writer is not None:
        writer.close()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()