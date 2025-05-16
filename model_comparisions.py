import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths to saved models
model_paths = {
    "Exp 1 (CPU 1T)"   : "resnet18_cifar10_cpu.pth",
    "Exp 2 (CPU 6T)"   : "resnet18_cifar10_cpu_mt.pth",
    "Exp 3 (GPU FP32)" : "resnet18_cifar10_gpu_fp32.pth",
    "Exp 5 (GPU AMP)"  : "resnet18_cifar10_gpu_amp.pth"
}

# CIFAR-10 test transform and dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])

batch_size = 128
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Accuracy storage
accuracies = {}

# Evaluation function
def evaluate(model_path, label):
    model = resnet18(weights=None, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Evaluating {label}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    acc = 100. * correct / total
    return acc

# Evaluate all models
for name, path in model_paths.items():
    try:
        acc = evaluate(path, name)
        accuracies[name] = acc
    except Exception as e:
        print(f"Failed to evaluate {name}: {e}")

# Plotting
plt.figure(figsize=(10,5))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison on CIFAR-10')
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

print("Model accuracy comparison complete. Plot saved as 'model_accuracy_comparison.png'.")