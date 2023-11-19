from differentiable_gate import *
from typing import List
from differentiable_circuit import CircuitChannel
from dataclasses import dataclass
from datatypes import *
import torch
from torch.nn import Parameter
import config
from torch import nn
import torch
from hamiltonian import Exp_i, Hamiltonian, HamiltonianTerm, TrotterSuzuki
import numpy as np
import copy
from datatypes import *


def convert(matrix):
    return torchcomplex(np.array(matrix))


"""Specific gates"""


@dataclass
class X(HamiltonianTerm):
    k = 1
    diag = False
    H = X = convert([[0, 1], [1, 0]])
    p: int


@dataclass
class Z(HamiltonianTerm):
    k = 1
    diag = True
    H = Z = convert([1, -1])
    p: int


@dataclass
class ZZ(HamiltonianTerm):
    k = 2
    diag = True
    H = ZZ = convert([1, -1, -1, 1])
    p: int
    q: int


@dataclass
class A(HamiltonianTerm):
    k = 2
    diag = False
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = XZ = convert(np.kron(X, Z))
    p: int
    q: int


@dataclass
class A2(HamiltonianTerm):
    k = 2
    diag = False
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    # H = XZ = convert(np.kron(X, Z))
    H = XX = convert(np.kron(X, X))
    p: int
    q: int


def bricklayer(n):
    l1 = list(range(0, n - 1, 2))
    l2 = list(range(1, n - 1, 2))
    return l1 + l2


def shift_right(H, d):
    H2 = copy.deepcopy(H)
    for h in H2.terms:
        if h.k == 1:
            h.p += d
        if h.k == 2:
            h.p += d
            h.q += d
    return H2


class TFIM(Hamiltonian):
    def __init__(self, n, coupling: float = 1.0):
        self.coupling = coupling
        self.Ising = [ZZ(i, i + 1) for i in bricklayer(n)]
        self.transverse = [X(i, strength=self.coupling) for i in range(n)]
        self.terms = self.Ising + self.transverse


class UnitaryBlock(CircuitChannel):
    def __init__(
        self,
        H: Hamiltonian,
        l: int = None,
        mixwith: List[int] = None,
        trottersteps: int = 1,
        reverse: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.H = H
        gates = []
        H_shifted = shift_right(H, 1)

        if mixwith is None:
            mixwith = [1] * l

        for i, mw in enumerate(mixwith):
            a = nn.Parameter(torch.randn(1))
            tau = nn.Parameter(torch.randn(1))
            zeta = nn.Parameter(torch.randn(1))

            step = [
                Exp_i(Z(0), a),
                TrotterSuzuki(H_shifted.Ising, H_shifted.transverse, tau, trottersteps),
                Exp_i(A(0, mw), zeta),
            ]
            if reverse:
                gates = gates + step[::-1]
            else:
                gates = gates + step

        self.gates = nn.ModuleList(gates)


class Block(CircuitChannel):
    def __init__(
        self,
        H: Hamiltonian,
        l: int = None,
        mixwith: List[int] = None,
        trottersteps: int = 1,
    ):
        nn.Module.__init__(self)
        A = AddAncilla(0)
        U = UnitaryBlock(H, l, mixwith, trottersteps)
        M = Measurement(0)
        self.gates = nn.ModuleList([A, U, M])


class ShortBlock(CircuitChannel):
    """In a short block we can apply the Hamiltonian to an n-qubit state
    instead of n+1 qubits.
    """

    def __init__(
        self,
        H: Hamiltonian,
        mixwith: int = 1,
        trottersteps: int = 1,
    ):
        a = nn.Parameter(torch.randn(1))
        tau = nn.Parameter(torch.randn(1))
        zeta = nn.Parameter(torch.randn(1))
        self.gates = nn.ModuleList(
            [
                TrotterSuzuki(H.Ising, H.transverse, tau, trottersteps),
                AddAncilla(0),
                Exp_i(Z(0), a),
                Exp_i(A(0, mixwith), zeta),
                Measurement(0),
            ]
        )


@dataclass
class StateGenerator:
    n: int

    def pure_state(self):
        raise NotImplementedError

    def density_matrix(self):
        raise NotImplementedError


@dataclass
class RandomState(StateGenerator):
    gen: torch.Generator


@dataclass
class HaarState(RandomState):
    def pure_state(self):
        slate = torch.zeros((2, 2**self.n), device=config.device)
        slate = slate.normal_(0, 1, generator=self.gen)
        x = torch.complex(slate[0], slate[1])
        x = x / torch.norm(x)
        return x

    def density_matrix(self):
        return torch.eye(2**self.n, device=config.device) / 2**self.n


@dataclass
class DensityMatrixState(RandomState):
    rho: DensityMatrix

    def pure_state(self):
        raise NotImplementedError

    def density_matrix(self):
        return self.rho


@dataclass
class PureState(StateGenerator):
    def density_matrix(self):
        psi = self.pure_state()
        return psi[:, None] * psi[None, :].conj()


@dataclass
class ZeroState(PureState):
    def pure_state(self):
        x = torch.zeros(2**self.n, device=config.device).to(tcomplex)
        x[0] = 1
        return x
