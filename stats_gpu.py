import torch

print("Checking GPU...")
try:
    # 1. Check visibility
    print(f"Is GPU available? {torch.cuda.is_available()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 2. The Stress Test (Matrix Math)
    x = torch.rand(5, 3).cuda()
    y = torch.rand(3, 4).cuda()
    z = torch.matmul(x, y)

    print("\n--- SUCCESS! ---")
    print("Your RTX 5060 is working correctly with PyTorch!")
    print(z)

except Exception as e:  
    print("\n--- ERROR ---")
    print(e)