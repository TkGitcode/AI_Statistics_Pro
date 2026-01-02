import torch
import matplotlib.pyplot as plt

# 1. Setup the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Topic 1 on: {device}")

# 2. Define Parameters
target_mean = 0.0   # Center of the bell
target_std = 1.0    # Width of the bell
n_samples = 10_000_000 # 10 Million points (Impossible for Excel, Easy for RTX 5060)

# 3. Generate Data
print(f"Generating {n_samples:,} samples...")
# torch.randn produces a "Standard Normal" (mean=0, std=1) automatically
data = torch.randn(n_samples, device=device)

# 4. Verify the Math (Statistics)
actual_mean = torch.mean(data).item()
actual_std = torch.std(data).item()

print("\n--- RESULTS ---")
print(f"Target Mean: {target_mean} | Actual Mean: {actual_mean:.5f}")
print(f"Target Std:  {target_std} | Actual Std:  {actual_std:.5f}")

# 5. Visualize (Histogram)
# We move a small subset to CPU just for plotting (plotting 10M points is slow/messy)
print("\nPlotting the histogram...")
data_cpu = data[:100000].cpu().numpy() # Take first 100k for the plot

plt.figure(figsize=(10, 6))
plt.hist(data_cpu, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f'Gaussian Distribution (Mean={target_mean}, Std={target_std})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.axvline(target_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig("01_gaussian_plot.png")
print("Graph saved as '01_gaussian_plot.png'")