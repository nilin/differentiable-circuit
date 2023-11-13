from differentiable_gate import *
from typing import List
from differentiable_circuit import Channel
from dataclasses import dataclass
from datatypes import *
import torch
import config
import torch
from hamiltonian import Exp_i, Hamiltonian, HamiltonianTerm
import numpy as np
from datatypes import *


def convert(matrix):
    return torchcomplex(np.array(matrix))


"""Specific gates"""


@dataclass
class X(HamiltonianTerm):
    k = 1
    diag = False
    H = X = convert([[0, 1], [1, 0]])


@dataclass
class ZZ(HamiltonianTerm):
    k = 2
    diag = True
    H = ZZ = convert([1, -1, -1, 1])


@dataclass
class A(HamiltonianTerm):
    k = 2
    diag = False
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = XZ = convert(np.kron(X, Z))


def bricklayer(a, b):
    """Assumes bricks of size 2.
    Input b is where the last brick ends
    (not where the last brick starts).
    """

    l1 = list(range(a, b - 1, 2))
    l2 = list(range(a + 1, b - 1, 2))
    return l1 + l2


class TFIM(Hamiltonian):
    def __init__(self, endpoints, coupling: float = 1.0):
        a, b = endpoints
        self.coupling = coupling
        self.Ising = [ZZ(i, i + 1) for i in bricklayer(a, b)]
        self.transverse = [X(i, strength=self.coupling) for i in range(a, b)]
        self.terms = self.Ising + self.transverse

    def TrotterSuzuki(self, tau: Scalar, steps: int):
        return super().TrotterSuzuki(self.transverse, self.Ising, tau, steps)


class Block(Channel):
    def __init__(
        self,
        n1,
        taus: List[Scalar],
        zetas: List[Scalar],
        trottersteps: int = 1,
        with_reset=True,
    ):
        tfim = TFIM((1, n1))
        self.gates = []
        for tau, zeta in zip(taus, zetas):
            self.gates += tfim.TrotterSuzuki(tau, trottersteps)
            self.gates.append(Exp_i(A(0, 1), zeta))

        if with_reset:
            self.gates.append(CleanSlateAncilla(0))


def zero_state(L):
    x = torch.zeros(2**L).to(tcomplex)
    x[0] = 1
    x = x.to(config.device)
    return x


def Haar_state(L, seed=0):
    N = 2**L
    x = torch.normal(0, 1, (2, N), generator=torch.Generator().manual_seed(seed))
    x = torch.complex(x[0], x[1]).to(tcomplex)
    x = x.to(config.device)
    x = x / torch.norm(x)
    return x
