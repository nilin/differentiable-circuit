import torch
from differentiable_gate import (
    Gate,
    Gate_1q,
    Gate_2q,
    Gate_2q_diag,
    Diag,
    GateState,
)
import numpy as np
from gate_implementations import torchcomplex
from dataclasses import dataclass


class ExpH(Gate):
    def __post_init__(self):
        self.compile()

    def compile(self):
        eigs, U = np.linalg.eigh(self.H)
        self.eigs = torchcomplex(eigs)
        self.U = torchcomplex(U)

    def control(self, t):
        D = torch.exp(-1j * t * self.eigs)
        return self.U @ (D[:, None] * self.U.T)


class ExpH_diag(ExpH, Diag):
    def compile(self):
        pass

    def control(self, t):
        return torch.exp(-1j * t * self.H)


def convert(matrix):
    return torchcomplex(np.array(matrix))



@dataclass
class UX(ExpH, Gate_1q):
    H = X = convert([[0, 1], [1, 0]])


@dataclass
class UZZ(ExpH_diag, Gate_2q_diag):
    H = ZZ = convert([1, -1, -1, 1])


@dataclass
class UA(ExpH, Gate_2q):
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = XZ = convert(np.kron(X, Z))


#def gradsafe_convert_to_tensor(gate_state):
#    """
#    We cannot apply torch.Tensor
#    to create the gate_state, as it will
#    not preserve the gradient.
#    """
#    if isinstance(gate_state, torch.Tensor):
#        return gate_state
#    else:
#        return torch.stack(
#            [gradsafe_convert_to_tensor(row) for row in gate_state]
#        )
#
#
#
# class UX(Gate_1q):
#    def control(self, t):
#        cos = torch.cos(t)
#        sin = torch.sin(t)
#        gate_state = [
#            [cos, -1j * sin],  # check if minus
#            [-1j * sin, cos],  #
#        ]
#        return gradsafe_convert_to_tensor(gate_state)
#
#
# class UZZ(Gate_2q_diag):
#    def control(self, t):
#        w = torch.exp(1j * t)
#        w_ = w.conj()
#        return gradsafe_convert_to_tensor([w_, w, w, w_])
#
#
# class UA(Gate_2q):
#    def control(self, t):
#        X = np.array([[0, 1], [1, 0]])
#        Z = np.array([[1, 0], [0, -1]])
#        XZ = np.kron(X, Z)
#        eigs, U = np.linalg.eigh(XZ)
#
#        eigs = torchcomplex(eigs)
#        U = torchcomplex(U)
#        D = torch.exp(-1j * t * eigs)
#
#        return U @ (D[:, None] * U.T)
#
