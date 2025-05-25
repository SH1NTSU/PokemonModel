import torch
print(torch.__version__)           # Should be ≥2.0.0
print(torch.cuda.is_available())   # Should be True
print(torch.version.cuda)          # Should match your driver (e.g., 12.1)


