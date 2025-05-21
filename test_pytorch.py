import torch

# Check PyTorch basics
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]])
print(f"Tensor:\n{x}")
