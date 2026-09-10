import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm   

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pt')
    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
        base_mae = MAE_ViT()
        base_mae.load_state_dict(state_dict)
        model = ViT_Classifier(base_mae.encoder, num_classes=10).to(device)
        writer = SummaryWriter(os.path.join('logs', 'cifar10', 'pretrain-cls'))
    else:
        base_mae = MAE_ViT()
        model = ViT_Classifier(base_mae.encoder, num_classes=10).to(device)
        writer = SummaryWriter(os.path.join('logs', 'cifar10', 'scratch-cls'))

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: (logit.argmax(dim=-1) == label).float().mean()

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
    step_count = 0
    optim.zero_grad()
    scaler = torch.amp.GradScaler("cuda")       
    for e in range(args.total_epoch):
        model.train()
        losses, acces = [], []
        for img, label in tqdm(train_dataloader, desc=f"epoch {e}"):
            step_count += 1
            with torch.amp.autocast("cuda"):           
                img, label = img.to(device), label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)

            scaler.scale(loss / steps_per_update).backward()

            if step_count % steps_per_update == 0:
                scaler.step(optim) 
                scaler.update()
                optim.zero_grad()

            losses.append(loss.item())
            acces.append(acc.item())

        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f"Epoch {e}: train_loss = {avg_train_loss:.4f}, train_acc = {avg_train_acc:.4f}")

        model.eval()
        val_losses, val_acces = [], []
        with torch.no_grad():
            for img, label in val_dataloader:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                val_losses.append(loss.item())
                val_acces.append(acc.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_acces) / len(val_acces)
        print(f"Epoch {e}: val_loss = {avg_val_loss:.4f}, val_acc = {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"-> saving checkpoint with val_acc = {best_val_acc:.4f} at epoch {e}")
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_acc': best_val_acc,
            }, args.output_model_path)

        writer.add_scalars('cls/loss', {'train': avg_train_loss, 'val': avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train': avg_train_acc, 'val': avg_val_acc}, global_step=e)