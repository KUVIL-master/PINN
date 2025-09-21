import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")

model = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
).to("cuda")

y = model(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")

# Result
# Input shape: torch.Size([2, 64, 16]), Output shape: torch.Size([2, 64, 16])