# AI Statistics & Probability (GPU Accelerated)

## Overview
This repository documents my MSc journey into the mathematical foundations of AI, implemented using **PyTorch on an NVIDIA RTX 5060**.
We use GPU acceleration to simulate millions of data points to verify statistical theorems empirically.

**Note** : *For GPU test in your device, here i have attached a stats_gpu.py file to Stress test the GPU.*

## Topics

### 1. The Normal (Gaussian) Distribution
- **File:** `Gaussian_Distribution/gaussian_simulation.py`
- **Concept:** The "Bell Curve" that describes errors and weights in AI.
- **Experiment:** Generated **10,000,000** samples on GPU.
- **Result:** Proof of Law of Large Numbers.

### 2. The Central Limit Theorem (CLT)
- **File:** `Central_Limit_Theorem/clt_gpu_simulation.py`
- **Concept:** Proving that the average of *any* random variables converges to a Gaussian distribution. This is why "Linear Regression" and "Neural Initialization" work.
- **Experiment:** Transformed a chaotic **Uniform Distribution** (10M points) into a perfect Bell Curve by averaging 10,000 experiments.
- **Result:** Successfully demonstrated convergence from Uniform to Gaussian distribution via simulation.

---
*Environment: PyTorch Nightly (CUDA 12.8)*