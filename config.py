import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = torch.Generator(device=device)


def randn():
    x = torch.randn(1, dtype=torch.float64, device=device, generator=gen)
    return torch.squeeze(x)
