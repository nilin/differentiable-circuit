from differentiable_gate import *
from non_unitary_gates import *
from typing import List
from differentiable_circuit import Circuit, UnitaryCircuit
from differentiable_channel import Channel
from dataclasses import dataclass
from datatypes import *
import torch
from torch.nn import Parameter
import config
from config import randn
from torch import nn
import torch
from hamiltonian import Exp_i, Hamiltonian, HamiltonianTerm, TrotterSuzuki
import numpy as np
import copy
from datatypes import *


def convert(matrix):
    return torchcomplex(torch.tensor(matrix).to(torch.complex64)).to(config.device)


"""Specific gates"""


class X(HamiltonianTerm):
    k = 1
    diag = False
    H = X = convert([[0, 1], [1, 0]])


class Z(HamiltonianTerm):
    k = 1
    diag = True
    H = Z = convert([1, -1])


class ZZ(HamiltonianTerm):
    k = 2
    diag = True
    H = ZZ = convert([1, -1, -1, 1])


class A(HamiltonianTerm):
    k = 2
    diag = False
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = XZ = convert(np.kron(X, Z))


def bricklayer(n):
    l1 = list(range(0, n - 1, 2))
    l2 = list(range(1, n - 1, 2))
    return l1 + l2


def shift_right(H, d):
    H2 = copy.deepcopy(H)
    for h in H2.terms:
        h.positions = tuple(p + d for p in h.positions)
        # if h.k == 1:
        #    h.p += d
        # if h.k == 2:
        #    h.p += d
        #    h.q += d
    return H2


class TFIM(Hamiltonian):
    def __init__(self, n, coupling: float = 1.0):
        self.coupling = coupling
        self.Ising = [ZZ(i, i + 1) for i in bricklayer(n)]
        self.transverse = [X(i).rescale(coupling) for i in range(n)]
        self.terms = self.Ising + self.transverse


class UnitaryBlock(UnitaryCircuit):
    def __init__(
        self,
        H: Hamiltonian,
        l: int = None,
        mixwith: List[int] = None,
        trottersteps: int = 1,
        use_trotter: bool = True,
        n: int = None,
        reverse: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.H = H
        gates = []

        if use_trotter:
            H_shifted = shift_right(H, 1)
        else:
            H_shifted = H.to_dense(n)
            H_shifted.ignore_positions = (0,)

        if mixwith is None:
            mixwith = [1] * l

        for i, mw in enumerate(mixwith):
            a = nn.Parameter(randn())
            tau = nn.Parameter(randn())
            zeta = nn.Parameter(randn())

            if use_trotter:
                e_iH = TrotterSuzuki(H_shifted.Ising, H_shifted.transverse, tau, trottersteps)
            else:
                e_iH = Exp_i(H_shifted, tau)

            step = [Exp_i(Z(0), a), e_iH, Exp_i(A(0, mw), zeta)]
            if reverse:
                gates = gates + step[::-1]
            else:
                gates = gates + step

        self.gates = nn.ModuleList(gates)


class Block(Channel):
    def __init__(
        self,
        H: Optional[Hamiltonian] = None,
        l: Optional[int] = None,
        mixwith: Optional[List[int]] = None,
        trottersteps: Optional[int] = 1,
        unitaryblock: Optional[UnitaryBlock] = None,
    ):
        nn.Module.__init__(self)
        A = AddAncilla(0)
        if unitaryblock is None:
            U = UnitaryBlock(H, l, mixwith, trottersteps)
        else:
            U = unitaryblock
        M = Measurement(0)
        self.gates = nn.ModuleList([A, U, M])


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
