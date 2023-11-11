import torch
from typing import Tuple, Callable, List, Iterable, Any

State = torch.Tensor
GateState = torch.Tensor
Scalar = torch.Tensor
ControlParams = Tuple[torch.Tensor]
uniform01 = float

"""density matrix used for testing"""
DensityMatrix = torch.Tensor


class GateImplementation:
    apply_gate: Callable[[Any, GateState, State], State]


def torchcomplex(x):
    real = torch.Tensor(x.real)
    imag = torch.Tensor(x.imag)
    return torch.complex(real, imag)
