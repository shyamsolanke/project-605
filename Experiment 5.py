import time
import os
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm  # Progress bar

# -- Experiment 5: GPU Single-GPU AMP (Mixed Precision) Setup --
print("\nInitializing Experiment 5: GPU Single-GPU AMP Setup... \n")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Hyperparameters
num_epochs    = 10
learning_rate = 0.001
batch_size    = 128

print(f"Configuration:\n"
      f"  Device        = {device}\n"
      f"  Epochs        = {num_epochs}\n"
      f"  Batch size    = {batch_size}\n"
      f"  Learning rate = {learning_rate}\n")

# Setup CPU process monitor
process = psutil.Process(os.getpid())

# CIFAR-10 transforms (same augmentations)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])

# DataLoader settings
loader_kwargs = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

print("Loading CIFAR-10 datasets...")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                    (0.247, 0.243, 0.261))
                                           ]))
test_loader  = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           prefetch_factor=2,
                                           persistent_workers=True)

print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples.\n")

# Model & AMP setup
print("Initializing ResNet-18 model, optimizer, and AMP scaler...")
model    = resnet18(weights=None, progress=True, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scaler    = GradScaler()
print(model, "\n")

# Profiling flag
enable_profiling = True

def train_epoch(epoch):
    print(f"--- Epoch {epoch} / {num_epochs} TRAINING ---")
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    data_time, comp_time = 0.0, 0.0
    start_epoch = time.perf_counter()
    last_time = start_epoch

    # Profile first epoch
    if enable_profiling and epoch == 1:
        print("Profiling first epoch on GPU (10 batches) with AMP...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i, (imgs, lbls) in enumerate(tqdm(train_loader, desc="Profiling", total=10)):
                t0 = time.perf_counter()
                data_time += t0 - last_time

                imgs, lbls = imgs.to(device), lbls.to(device)
                comp_start = time.perf_counter()

                optimizer.zero_grad()
                with autocast():
                    out  = model(imgs)
                    loss = criterion(out, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                comp_time += time.perf_counter() - comp_start
                last_time = time.perf_counter()

                running_loss += loss.item() * lbls.size(0)
                _, preds = out.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)
                if i >= 9:
                    break

        summary = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print("Profiler summary (top GPU ops):\n", summary)
        prof.export_chrome_trace("profiler_exp5_trace.json")
        with open("profiler_exp5_summary.txt", "w") as f:
            f.write(summary)
        print("Saved profiler outputs.\n")

    # Normal training loop
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="batch"):
        bs = time.perf_counter()
        data_time += bs - last_time

        imgs, lbls = imgs.to(device), lbls.to(device)
        comp_start = time.perf_counter()

        optimizer.zero_grad()
        with autocast():
            out  = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        comp_time += time.perf_counter() - comp_start

        running_loss += loss.item() * lbls.size(0)
        _, preds = out.max(1)
        correct += preds.eq(lbls).sum().item()
        total += lbls.size(0)

        last_time = time.perf_counter()

    epoch_time = time.perf_counter() - start_epoch
    throughput  = total / epoch_time
    avg_data    = data_time / (total / batch_size)
    avg_comp    = comp_time / (total / batch_size)
    acc         = 100. * correct / total
    torch_mem   = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(f"--> Epoch {epoch} | Loss: {running_loss/total:.4f} | Acc: {acc:.2f}%")
    print(f"    Time: {epoch_time:.2f}s | Throughput: {throughput:.2f} img/s")
    print(f"    Data avg: {avg_data:.4f}s | Comp avg: {avg_comp:.4f}s")
    print(f"    GPU mem (MB): {torch_mem:.2f}\n")


def eval_model():
    print("--- EVALUATION ---")
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    start = time.perf_counter()

    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Eval", unit="batch"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            with autocast():
                out  = model(imgs)
                loss = criterion(out, lbls)

            running_loss += loss.item() * lbls.size(0)
            _, preds = out.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)

    elapsed = time.perf_counter() - start
    acc     = 100. * correct / total
    print(f"--> Test | Loss: {running_loss/total:.4f} | Acc: {acc:.2f}% | Time: {elapsed:.2f}s")

    ckpt_name = 'resnet18_cifar10_gpu_amp.pth'
    torch.save(model.state_dict(), ckpt_name)
    print(f"Saved checkpoint: {ckpt_name}\n")


if __name__ == '__main__':
    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch)
        eval_model()
    print("All done. Experiment 5 complete.")
