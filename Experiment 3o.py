import time
import os
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.profiler import profile, ProfilerActivity, record_function
from tqdm import tqdm  # Progress bar

# -- Experiment 3: GPU Single-GPU Optimized Setup --
print("\nInitializing Experiment 3: GPU Single-GPU Optimized Setup...\n")

# Device & tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
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

# CPU process monitor
process = psutil.Process(os.getpid())

# Mixed precision scaler
scaler = GradScaler()

# Data loaders
def get_dataloaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type=='cuda'),
        prefetch_factor=2,
        persistent_workers=True
    )
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(
        test_set, shuffle=False, **loader_kwargs)
    print(f"Datasets: {len(train_set)} train, {len(test_set)} test samples\n")
    return train_loader, test_loader

train_loader, test_loader = get_dataloaders(batch_size)

# Model, loss, optimizer
model = resnet18(weights=None, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
print(model, "\n")

# Profiling flag
enable_profiling = True

# Utility to sync and time

def sync_time():
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()

# Training epoch
def train_epoch(epoch):
    print(f"--- Epoch {epoch}/{num_epochs} TRAIN ---")
    model.train()
    start_epoch = time.perf_counter()
    data_time = 0.0
    comp_time = 0.0
    last = start_epoch
    running_loss = correct = total = 0

    # Profile first batches
    if enable_profiling and epoch == 1:
        print("Profiling first 10 batches (CPU+CUDA)...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i, (imgs, labels) in enumerate(
                    tqdm(train_loader, desc="Profiling batches", total=10)):
                # data timing
                t0 = sync_time()
                data_time += t0 - last
                # transfer
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # forward+backward
                with record_function("forward_backward"): 
                    with autocast(device_type=device.type):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                # compute timing
                t1 = sync_time()
                comp_time += t1 - t0
                last = t1
                # stats
                running_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                if i >= 9:
                    break
        # show profiler summary sorted by cuda time
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        prof.export_chrome_trace("exp3_optimized_trace.json")
        print("Profiler trace saved.\n")

    # Full epoch training
    for imgs, labels in tqdm(train_loader, desc="Training", unit="batch"):
        t0 = sync_time()
        data_time += t0 - last
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with record_function("forward_backward_batch"):
            with autocast(device_type=device.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        t1 = sync_time()
        comp_time += t1 - t0
        last = t1
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_time = time.perf_counter() - start_epoch
    acc = 100. * correct / total
    throughput = total / epoch_time
    print(f"Epoch {epoch} done | Loss: {running_loss/total:.4f} | "
          f"Acc: {acc:.2f}% | Time: {epoch_time:.2f}s | "
          f"Throughput: {throughput:.2f} img/s")
    print(f" Data load time: {data_time:.3f}s | compute time: {comp_time:.3f}s\n")

# Evaluation
def eval_model():
    print("--- Evaluation ---")
    model.eval()
    start = sync_time()
    running_loss = correct = total = 0
    for imgs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            with autocast(device_type=device.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    eval_time = sync_time() - start
    acc = 100. * correct / total
    print(f"Eval | Loss: {running_loss/total:.4f} | Acc: {acc:.2f}% | Time: {eval_time:.2f}s\n")
    torch.save(model.state_dict(), 'resnet18_cifar10_gpu_optimized.pth')
    print("Checkpoint saved: resnet18_cifar10_gpu_optimized.pth\n")

if __name__ == '__main__':
    for epoch in range(1, num_epochs+1):
        train_epoch(epoch)
        eval_model()
    print("All done. Experiment 3 optimized complete.")
