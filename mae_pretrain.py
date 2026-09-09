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
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model import MAE_ViT, rearrange
from utils import setup_seed

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--per_gpu_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    args = parser.parse_args()

    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup_seed(args.seed + global_rank)
    torch.backends.cudnn.benchmark = True

    # Batch calculation across GPUs
    total_device_batch = args.per_gpu_batch_size * world_size # 512 * 2 = 1024
    assert args.batch_size % total_device_batch == 0
    steps_per_update = args.batch_size // total_device_batch # 4096 // 1024 = 4 steps

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=(global_rank == 0), transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=(global_rank == 0), transform=transform)
    
    # Synchronize so rank 1 doesn't read data while rank 0 is downloading
    torch.distributed.barrier()

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain')) if global_rank == 0 else None

    # Model & DDP wrapper
    model = MAE_ViT(mask_ratio=args.mask_ratio).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

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
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    scaler = GradScaler('cuda')

    # Fixed validation sample cached on rank 0
    if global_rank == 0:
        fixed_val_img = torch.stack([val_dataset[i][0] for i in range(16)]).to(local_rank)

    step_count = 0
    optim.zero_grad(set_to_none=True)

    for e in range(args.total_epoch):
        train_sampler.set_epoch(e)
        model.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"epoch {e}") if global_rank == 0 else dataloader

        for img, _ in pbar:
            step_count += 1
            img = img.to(local_rank, non_blocking=True)

            # Mixed precision execution
            with autocast(device_type='cuda', dtype=torch.float16):
                predicted_img, mask = model(img)
                loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                loss_scaled = loss / steps_per_update

            scaler.scale(loss_scaled).backward()

            if step_count % steps_per_update == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running_loss += loss.detach()

        lr_scheduler.step()

        # Metrics and visualization only on master rank
        if global_rank == 0:
            avg_loss = (running_loss / len(dataloader)).item()
            writer.add_scalar('train/loss', avg_loss, global_step=e)
            writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step=e)
            print(f'Epoch {e}: train loss = {avg_loss:.5f}')

            if (e + 1) % 50 == 0 or e == args.total_epoch - 1:
                model.eval()
                with torch.no_grad():
                    with autocast(device_type='cuda', dtype=torch.float16):
                        raw_model = model.module
                        pred_val, mask = raw_model(fixed_val_img)
                        pred_val = pred_val * mask + fixed_val_img * (1 - mask)

                    vis = torch.cat([fixed_val_img * (1 - mask), pred_val, fixed_val_img], dim=0)
                    vis = rearrange(vis, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                    writer.add_image('val/reconstructions', (vis + 1) / 2, global_step=e)

            if (e + 1) % 100 == 0 or e == args.total_epoch - 1:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': avg_loss,
                }, args.model_path)

    if global_rank == 0:
        writer.close()
    destroy_process_group()

if __name__ == '__main__':
    main()