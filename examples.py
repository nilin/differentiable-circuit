from differentiable_gate import Scalar, Gate, CleanSlateAncilla
from typing import Callable, List
from differentiable_circuit import Circuit, Params, overlap, State, Channel
from dataclasses import dataclass
import torch
import config
import torch
from differentiable_gate import (
    Gate_1q,
    Gate_2q,
    Gate_2q_diag,
)
from Stones_theorem import Exp_iH, Exp_iH_diag, Hamiltonian
import numpy as np
from gate_implementations import torchcomplex


def convert(matrix):
    return torchcomplex(np.array(matrix))


"""Specific gates"""


@dataclass
class UX(Exp_iH, Gate_1q):
    H = X = convert([[0, 1], [1, 0]])


@dataclass
class UZZ(Exp_iH_diag, Gate_2q_diag):
    H = ZZ = convert([1, -1, -1, 1])


@dataclass
class UA(Exp_iH, Gate_2q):
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = XZ = convert(np.kron(X, Z))


class TFIM(Hamiltonian):
    def __init__(self, endpoints, coupling: float = 1.0):
        a, b = endpoints
        self.coupling = coupling
        self.Ising = [UZZ(i, i + 1) for i in range(a, b - 1)]
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
