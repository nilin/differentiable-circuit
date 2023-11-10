from differentiable_gate import Scalar, Gate
from gates import ExpH
from differentiable_circuit import UnitaryCircuit, Params, overlap
import gates
from typing import Callable, List
import torch
import config


def TrotterSuzuki(
    Layer1: List[ExpH],
    Layer2: List[ExpH],
    T1: Scalar,
    T2: Scalar,
    k: int,
):
    t1 = T1 / k
    t2 = T2 / k
    U_0 = [U.set_input(t2 / 2) for U in Layer2]
    U_1 = [U.set_input(t1) for U in Layer1]
    U_2 = [U.set_input(t2) for U in Layer2]
    return U_0 + (U_1 + U_2) * (k - 1) + U_1 + U_0


class Block(UnitaryCircuit):
    def __init__(self, L, k, coupling: float, T: Scalar, zeta: Scalar):
        UZZs = [gates.UZZ(i, i + 1) for i in range(L - 1)]
        UXs = [gates.UX(i) for i in range(L)]
        UA = gates.UA(0, 1, input=zeta)
        self.gates = TrotterSuzuki(UZZs, UXs, -T, -coupling * T, k) + [UA]


def zero_state(L):
    x = torch.zeros(2**L).to(config.tcomplex)
    x[0] = 1
    x = x.to(config.device)
    return x


def Haar_state(L, seed=0):
    N = 2**L
    x = torch.normal(0, 1, (2, N), generator=torch.Generator().manual_seed(seed))
    x = torch.complex(x[0], x[1]).to(config.tcomplex)
    x = x.to(config.device)
    x = x / torch.norm(x)
    return x
