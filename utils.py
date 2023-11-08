from collections import namedtuple
import torch
import numpy as np

Regs = namedtuple("Regs", ["psi", "scratch"])


def torchcomplex(x):
    real = torch.Tensor(x.real)
    imag = torch.Tensor(x.imag)
    return torch.complex(real, imag)
