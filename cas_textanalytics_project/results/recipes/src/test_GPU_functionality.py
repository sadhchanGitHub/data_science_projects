import torch

# Test tensor creation on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.rand(3, 3).to(device)
print("Tensor on:", device)
print(tensor)
