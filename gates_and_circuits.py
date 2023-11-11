from differentiable_gate import Scalar, Gate
from typing import Callable, List
from differentiable_circuit import Circuit, Params, cdot, State, Channel
from dataclasses import dataclass
import torch
import config
import torch
from Stones_theorem import Exp_iH, Hamiltonian
import numpy as np
from gate_implementation import torchcomplex

uniform01 = float


def convert(matrix):
    return torchcomplex(np.array(matrix))


"""Specific gates"""


@dataclass
class UX(Exp_iH):
    k = 1
    diag = False
    H = X = convert([[0, 1], [1, 0]])


@dataclass
class UZZ(Exp_iH):
    k = 2
    diag = True
    H = ZZ = convert([1, -1, -1, 1])


@dataclass
class UA(Exp_iH):
    k = 2
    diag = False
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = XZ = convert(np.kron(X, Z))


def bricklayer(a, b):
    """Assumes bricks of size 2. b is the rightmost endpoint (i+1<b)."""
    l1 = list(range(a, b - 1, 2))
    l2 = list(range(a + 1, b - 1, 2))
    return l1 + l2


class TFIM(Hamiltonian):
    def __init__(self, endpoints, coupling: float = 1.0):
        a, b = endpoints
        self.coupling = coupling
        self.Ising = [UZZ(i, i + 1) for i in bricklayer(a, b)]
        self.transverse = [UX(i, strength=self.coupling) for i in range(a, b)]
        self.terms = self.Ising + self.transverse

    def TrotterSuzuki(self, tau: Scalar, steps: int):
        return super().TrotterSuzuki(self.transverse, self.Ising, tau, steps)


class Block(Circuit):
    def __init__(self, L, tau: Scalar, zeta: Scalar, trottersteps: int = 2):
        tfim = TFIM((1, L))
        self.gates = tfim.TrotterSuzuki(tau, trottersteps) + [UA(0, 1, input=zeta)]


class Lindblad(Channel):
    def __init__(self, *blocks):
        self.blocks = blocks
        self.measurements = [CleanSlateAncilla()] * len(self.blocks)


"""Non-unitary gates"""


class Measurement(Gate):
    def apply(self, x: State, u: uniform01):
        N = len(x)
        p0 = probabilitymass(x[: N // 2]) / probabilitymass(x)

        outcome = u > p0
        return self.cut(x, outcome), outcome, (p0 if outcome == 0 else 1 - p0)

    def cut(self, x: State, outcome: bool):
        N = len(x)
        if outcome:
            return x[N // 2 :]
        else:
            return x[: N // 2]

    def embed(self, x: State, outcome: bool):
        if outcome:
            return torch.cat((torch.zeros_like(x), x))
        else:
            return torch.cat((x, torch.zeros_like(x)))

    def reverse(self, x: State, outcome: bool):
        return self.embed(x, outcome)


def probabilitymass(x):
    return torch.sum(torch.abs(x) ** 2).real


class CleanSlateAncilla(Measurement):
    def apply(self, x: State, u: uniform01):
        N = len(x)
        p0 = probabilitymass(x[: N // 2]) / probabilitymass(x)

        outcome = u > p0
        x = self.embed(self.cut(x, outcome), 0)
        return x, outcome, (p0 if outcome == 0 else 1 - p0)

    def reverse(self, x: State, outcome: bool):
        return self.embed(self.cut(x, 0), outcome)


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
