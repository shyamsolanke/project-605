import time
import os
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm  # Progress bar

# -- Experiment 1: CPU Single-Thread Setup --
print("\nInitializing Experiment 1: CPU Single-Thread Setup... \n")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device("cpu")

# Hyperparameters
num_epochs    = 10
learning_rate = 0.001
batch_size    = 128

# Print configuration
print(f"Configuration:\n"
      f"  Device        = {device}\n"
      f"  Threads       = {torch.get_num_threads()}\n"
      f"  Epochs        = {num_epochs}\n"
      f"  Batch size    = {batch_size}\n"
      f"  Learning rate = {learning_rate}\n")

# Setup process for monitoring
process = psutil.Process(os.getpid())

# CIFAR-10 data transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])

# Datasets & loaders
print("Loading CIFAR-10 datasets...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=0, pin_memory=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True,
    transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size,
    shuffle=False, num_workers=0, pin_memory=False)

print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples.\n")

# Model and optimizer setup
print("Initializing model and optimizer...")
model = resnet18(
    weights=None,
    progress=True,
    num_classes=10
).to(device)
print(f"Model architecture:\n{model}\n")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate,
    momentum=0.9, weight_decay=5e-4
)
print("Model and optimizer initialized.\n")

# Profiling flag
enable_profiling = True


def train_epoch(epoch):
    print(f"--- Starting Epoch {epoch} Training ---")
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    data_time, comp_time = 0.0, 0.0
    start_epoch = time.perf_counter()
    last_time = start_epoch

    # Optionally run one epoch under profiler
    if enable_profiling and epoch == 1:
        print("Starting PyTorch Profiler for first epoch (10 batches)...")
        with profile(
            activities=[ProfilerActivity.CPU]
        ) as prof:
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Profiling", total=10)):
                t0 = time.perf_counter()
                data_time += t0 - last_time
                images, labels = images.to(device), labels.to(device)
                comp_start = time.perf_counter()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                comp_time += time.perf_counter() - comp_start
                last_time = time.perf_counter()
                running_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                if batch_idx >= 9:
                    break
        summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        print("Profiler summary (top 10):\n", summary)
        with open("profiler_epoch1_summary.txt", "w") as f:
            f.write(summary)
        prof.export_chrome_trace("profiler_epoch1_trace.json")
        print("Profiler results saved to files.\n")

    # Continue normal training
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="batch")):
        batch_start = time.perf_counter()
        data_time += batch_start - last_time
        images, labels = images.to(device), labels.to(device)
        comp_start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        comp_time += time.perf_counter() - comp_start
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        last_time = time.perf_counter()

    epoch_time = time.perf_counter() - start_epoch
    avg_data = data_time / (total / batch_size)
    avg_comp = comp_time / (total / batch_size)
    acc = 100. * correct / total
    print(f"--> Epoch {epoch} TRAIN Complete | Loss: {running_loss/total:.4f} | "
          f"Acc: {acc:.2f}% | Time: {epoch_time:.2f}s | "
          f"Data avg: {avg_data:.4f}s | Comp avg: {avg_comp:.4f}s\n")


def eval_model():
    print("--- Starting Evaluation ---")
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluation", unit="batch")):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    elapsed = time.perf_counter() - start
    acc = 100. * correct / total
    print(f"--> Test Complete | Loss: {running_loss/total:.4f} | "
          f"Acc: {acc:.2f}% | Time: {elapsed:.2f}s")
    print("Saving model checkpoint...")
    torch.save(model.state_dict(), 'resnet18_cifar10_cpu.pth')
    print("Model saved as resnet18_cifar10_cpu.pth\n")

if __name__ == '__main__':
    for epoch in range(1, num_epochs+1):
        train_epoch(epoch)
        eval_model()
    print("All done. Experiment complete.")
