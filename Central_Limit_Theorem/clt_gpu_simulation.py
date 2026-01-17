import torch
import matplotlib.pyplot as plt
import time
import os

# 1. Setup Device (RTX 5060)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def run_clt_simulation():
    # Parameters
    NUM_EXPERIMENTS = 10000   # How many "means" we want to calculate
    SAMPLE_SIZE = 1000        # How many numbers per experiment
    TOTAL_POINTS = NUM_EXPERIMENTS * SAMPLE_SIZE

    print(f"Generating {TOTAL_POINTS:,} points from a UNIFORM distribution...")

    # 2. Generate Non-Normal Data (Uniform Distribution)
    # This data is completely flat. No bell curve here.
    start_time = time.time()
    
    # Create a tensor of shape [10000, 1000]
    # uniform_data ~ U(0, 1)
    uniform_data = torch.rand((NUM_EXPERIMENTS, SAMPLE_SIZE), device=device)
    
    gpu_time = time.time() - start_time
    print(f"Data Generation Time: {gpu_time:.4f} seconds")

    # 3. Apply Central Limit Theorem
    # Calculate the mean across the sample dimension (dim=1)
    # We are collapsing [10000, 1000] -> [10000]
    sample_means = torch.mean(uniform_data, dim=1)

    # Move to CPU for plotting (Matplotlib doesn't support GPU tensors directly)
    sample_means_cpu = sample_means.cpu().numpy()
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))

    # Subplot 1: The Source (A subset of raw uniform data)
    plt.subplot(1, 2, 1)
    # Take first 10000 points just to show the shape
    flat_data = uniform_data.view(-1)[:10000].cpu().numpy() 
    plt.hist(flat_data, bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.title("Origin: Uniform Distribution (Flat)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Subplot 2: The Result (The Sample Means)
    plt.subplot(1, 2, 2)
    plt.hist(sample_means_cpu, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.title(f"Result: Sampling Distribution (Gaussian!)\nn={SAMPLE_SIZE}")
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")

    # Theoretical Mean Calculation
    # For U(0,1), Mean is 0.5. Our peak should be at 0.5.
    plt.axvline(0.5, color='red', linestyle='dashed', linewidth=1, label='Theoretical Mean (0.5)')
    plt.legend()

    plt.tight_layout()
    
    # Save plot instead of showing (Professional workflow)
    output_path = "02_Central_Limit_Theorem/clt_proof.png"
    # Ensure directory exists if running from root
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Simulation Complete. Plot saved to {output_path}")

if __name__ == "__main__":
    run_clt_simulation()