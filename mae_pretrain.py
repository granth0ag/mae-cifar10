import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import MAE_ViT, rearrange
from utils import setup_seed

if __name__ == '__main__':
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

    setup_seed(args.seed)

    batch_size = args.batch_size    
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)

    # linear scaling rule: lr = base_lr * batch_size / 256
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.base_learning_rate * args.batch_size / 256, 
        betas=(0.9, 0.95), 
        weight_decay=args.weight_decay
    )

    # warmup + cosine decay
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epoch + 1e-8), 
        0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    step_count = 0
    optim.zero_grad()

    scaler = torch.cuda.amp.GradScaler()

    for e in range(args.total_epoch):
        model.train()
        losses = []

        for img, _ in tqdm(dataloader, desc=f"epoch {e}"):
            step_count += 1
            img = img.to(device)

            with torch.cuda.amp.autocast():
                predicted_img, mask = model(img)
                loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            scaler.scale(loss / steps_per_update).backward()

            if step_count % steps_per_update == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            
            predicted_img, mask = model(img)
            # compute loss only on masked patches
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            (loss / steps_per_update).backward()

            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()

            losses.append(loss.item())

        lr_scheduler.step()      
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('train/loss', avg_loss, global_step=e)
        writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step=e)
        print(f'Epoch {e}: train loss = {avg_loss:.5f}')

        # validation reconstruction
        if e % 10 == 0 or e == args.total_epoch - 1:
            model.eval()
            with torch.no_grad():
                val_img = torch.stack([val_dataset[i][0] for i in range(16)]).to(device)
                predicted_val_img, mask = model(val_img)
                predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)

                vis = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
                vis = rearrange(vis, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                writer.add_image('val/reconstructions', (vis + 1) / 2, global_step=e)

        if e % 10 == 0 or e == args.total_epoch - 1:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': avg_loss,
            }, args.model_path)
    writer.close()