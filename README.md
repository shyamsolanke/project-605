# 🚀 ResNet-18 Performance Profiling on CIFAR-10

This project analyzes and optimizes the training performance of **ResNet-18** on the **CIFAR-10** dataset using **PyTorch**. It involves profiling and benchmarking across multiple hardware setups, including CPU (single and multi-threaded), GPU (FP32), and GPU with **Automatic Mixed Precision (AMP)**.

---

## 📌 Project Objectives

- Identify computational and memory bottlenecks during training  
- Measure and compare training **throughput**, **epoch time**, and **GPU memory usage**  
- Utilize `torch.profiler` for kernel-level insights  
- Evaluate impact of **AMP** and **batch size tuning** on performance  
- Optimize training pipeline step-by-step from CPU to GPU

---

## 🧪 Experiments Conducted

| Experiment | Setup                         | Key Features                       |
|------------|-------------------------------|------------------------------------|
| **Exp 1**  | CPU (1 Thread)                | Baseline, no parallelism           |
| **Exp 2**  | CPU (6 Threads)               | Multi-threaded + DataLoader workers |
| **Exp 3**  | GPU (FP32)                    | Full precision on CUDA device      |
| **Exp 4**  | GPU (AMP) w/ Batch Sweep      | Batch sizes from 16 to 512         |
| **Exp 5**  | GPU (AMP, fixed batch)        | Mixed precision + GradScaler       |

---

## 📊 Key Results Summary

| Metric               | Exp 1 (CPU 1T) | Exp 2 (CPU 6T) | Exp 3 (GPU FP32) | Exp 5 (GPU AMP) |
|----------------------|----------------|----------------|------------------|-----------------|
| Epoch Time           | ~19 s          | ~7.6 s         | ~2.0 s           | ~1.8 s          |
| Throughput (img/s)   | ~180           | ~380           | ~5200            | ~6800           |
| Accuracy             | ~43%           | ~65%           | ~66%             | ~66%            |
| GPU Mem Usage        | N/A            | N/A            | ~950 MB          | ~925 MB         |

---

## 📦 Project Structure


---

## 🧠 Key Concepts

- **autocast()**: Automatically chooses FP16 for safe operations and FP32 for sensitive ones.
- **GradScaler()**: Dynamically scales gradients to avoid underflow in FP16 training.
- **torch.profiler**: Tracks execution time per operator and helps locate bottlenecks.

---

## 📈 Insights

- GPU + AMP achieved **35× speedup** over single-threaded CPU.
- AMP reduced GPU memory usage while preserving accuracy.
- Batch size tuning revealed **256** as the sweet spot for performance vs memory.
- `convolution_backward` was the most time-consuming kernel in all experiments.

---

## ✅ Requirements

- Python 3.8+  
- PyTorch 1.13+  
- CUDA-enabled GPU (for GPU experiments)  
- tqdm, matplotlib, psutil

```bash
pip install torch torchvision matplotlib tqdm psutil

