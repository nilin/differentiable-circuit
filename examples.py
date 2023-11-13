from differentiable_gate import *
from typing import Callable, List
from differentiable_circuit import Circuit, Channel
from dataclasses import dataclass
import torch
import config
import torch
from torch.nn import Parameter, ParameterList, ParameterDict
from hamiltonian import Exp_i, Hamiltonian, HamiltonianTerm
import numpy as np
from collections import namedtuple
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


class Block(Circuit):
    def __init__(self, n1, tau: Scalar, zeta: Scalar, trottersteps: int = 2):
        tfim = TFIM((1, n1))
        self.gates = tfim.TrotterSuzuki(tau, trottersteps) + [Exp_i(A(0, 1), zeta)]


class Lindblad(Channel):
    def __init__(self, n, l, trottersteps: int = 2):
        self.gates = []
        tfim = TFIM((0, n))

        self.parameters = ParameterDict(
            dict(
                taus=Parameter(torch.tensor([1.0] * l)),
                zetas=Parameter(torch.tensor([1.0] * l)),
            )
        )

        for tau, zeta in zip(self.parameters.taus, self.parameters.zetas):
            self.gates += tfim.TrotterSuzuki(tau, trottersteps)
            self.gates += [Add_0_Qubit()]
            self.gates += [Exp_i(A(0, 1), zeta)]
            self.gates += [Measurement(0)]


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
