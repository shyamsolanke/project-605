import time
import os
import psutil
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    # -- Experiment 4: GPU Mixed‑Precision Batch‑size Sweep with Plots & Output Saving --
    print("\nStarting Experiment 4: Batch‑size Sweep with AMP on GPU…\n")

    # Device & cuDNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}\n")

    # Sweep settings
    batch_sizes = [16, 32, 64, 128, 256, 512]
    num_epochs   = 1  # single epoch per batch size for sweep metrics
    learning_rate = 0.001

    # CPU monitor
    process = psutil.Process(os.getpid())

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
    ])

    # Load datasets once
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Prepare CSV output file
    data_file = 'exp4_results.csv'
    with open(data_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['batch_size','throughput_img_per_s','peak_gpu_mem_MB'])

    # Metric storage
    throughputs = []
    gpu_mems = []

    for batch_size in batch_sizes:
        print(f"===== Batch size: {batch_size} =====")
        # DataLoader
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)

        # Model, loss, optimizer, scaler
        model = resnet18(weights=None, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        scaler = GradScaler()

        # Single epoch for sweep metrics
        model.train()
        total_samples = 0
        start_time = time.perf_counter()

        for imgs, labels in tqdm(train_loader, desc=f"Training bs={batch_size}", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_samples += labels.size(0)

        epoch_time = time.perf_counter() - start_time
        throughput = total_samples / epoch_time
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

        print(f"Batch {batch_size} | Time: {epoch_time:.2f}s | Throughput: {throughput:.2f} img/s | Mem: {peak_mem:.2f} MB")

        throughputs.append(throughput)
        gpu_mems.append(peak_mem)

        # Append to CSV
        with open(data_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([batch_size, f"{throughput:.2f}", f"{peak_mem:.2f}"])

        # cleanup
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Plot throughput vs batch size
    plt.figure()
    plt.plot(batch_sizes, throughputs, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (img/s)')
    plt.title('Experiment 4: Throughput vs Batch Size')
    plt.grid(True)
    plt.savefig('throughput_vs_batch.png')
    plt.show()

    # Plot GPU memory vs batch size
    plt.figure()
    plt.plot(batch_sizes, gpu_mems, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Peak GPU Memory (MB)')
    plt.title('Experiment 4: GPU Memory vs Batch Size')
    plt.grid(True)
    plt.savefig('gpu_memory_vs_batch.png')
    plt.show()

    print(f"Results written to {data_file}, plots saved as 'throughput_vs_batch.png' and 'gpu_memory_vs_batch.png'.")


if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
