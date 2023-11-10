import torch
from differentiable_gate import (
    Gate,
    Gate_1q,
    Gate_2q,
    Gate_2q_diag,
    Diag,
    Measurement,
    State,
)
import numpy as np
from gate_implementations import torchcomplex
from dataclasses import dataclass


"""Define gates as Hamiltonian evolution"""


class Exp_iH(Gate):
    def __post_init__(self):
        self.compile()

    def compile(self):
        eigs, U = np.linalg.eigh(self.H)
        self.eigs = torchcomplex(eigs)
        self.U = torchcomplex(U)

    def control(self, t):
        D = torch.exp(-1j * t * self.eigs)
        return self.U @ (D[:, None] * self.U.T)


class Exp_iH_diag(Exp_iH, Diag):
    def compile(self):
        pass

    def control(self, t):
        return torch.exp(-1j * t * self.H)


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
