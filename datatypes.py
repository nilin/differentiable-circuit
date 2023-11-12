import torch
from typing import Tuple, Callable, List, Iterable, Any

State = torch.Tensor
GateState = torch.Tensor
Scalar = torch.Tensor
ControlParams = Tuple[torch.Tensor]
uniform01 = float
ignore = Any

"""density matrix used for testing"""
DensityMatrix = torch.Tensor


class GateImplementation:
    apply_gate: Callable[[Any, GateState, State], State]


def torchcomplex(x):
    real = torch.Tensor(x.real)
    imag = torch.Tensor(x.imag)
    return torch.complex(real, imag)


def cdot(phi, psi):
    return phi.conj().dot(psi)


def squared_overlap(phi, psi):
    return torch.abs(cdot(phi, psi)) ** 2


def probabilitymass(x):
    return torch.sum(torch.abs(x) ** 2).real
