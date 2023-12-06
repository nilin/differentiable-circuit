from .gate import GateState, Scalar, Gate, ThetaGate, Op
from .circuit import Non_unitary_circuit
from . import gate
from typing import List
from torch import nn
from . import config
from .config import randn
from copy import deepcopy
from dataclasses import dataclass, KW_ONLY
import torch
from .gate import (
    Gate,
    State,
    DenseGate,
)
import numpy as np
from .datatypes import *


"""Define gates as Hamiltonian evolution"""


class HamiltonianTerm(Gate):
    H: GateState

    def __init__(self, *positions):
        Gate.__init__(self, positions)

    def apply(self, psi: State):
        return self.apply_gate_state(self.H, psi)

    def rescale(self, c: float):
        self = deepcopy(self)
        self.H = c * self.H
        return self


@dataclass
class DenseHamiltonian(HamiltonianTerm, DenseGate):
    diag = False
    H: GateState


@dataclass
class Hamiltonian(Op):
    terms = List[HamiltonianTerm]

    def apply(self, psi: State):
        H_psi = torch.zeros_like(psi)
        for H_i in self.terms:
            H_psi += H_i.apply(psi)
        return H_psi

    def expectation(self, psi: State):
        return psi.conj().dot(self.apply(psi))

    """
    Test utilities.
    """

    def to_dense(self, n):
        H = self.create_dense_matrix(n)
        return DenseHamiltonian(H)


class TrotterSuzuki(Non_unitary_circuit):
    Layer1: List[HamiltonianTerm]
    Layer2: List[HamiltonianTerm]
    T: Scalar
    steps: int

    def __init__(self, Layer1, Layer2, T=None, steps=1):
        torch.nn.Module.__init__(self)
        self.Layer1 = Layer1
        self.Layer2 = Layer2
        self.steps = steps
        if T is None:
            self.T = nn.Parameter(randn())
        else:
            self.T = T

        U_0 = [Exp_i(H, self.T, speed=1 / (2 * self.steps)) for H in self.Layer2]
        U_1 = [Exp_i(H, self.T, speed=1 / self.steps) for H in self.Layer1]
        U_2 = [Exp_i(H, self.T, speed=1 / self.steps) for H in self.Layer2]
        self.gates = nn.ModuleList(U_0 + (U_1 + U_2) * (self.steps - 1) + U_1 + U_0)


class Exp_i(ThetaGate, nn.Module):
    hamiltonian: HamiltonianTerm = None

    def __init__(self, hamiltonian: HamiltonianTerm, T: Scalar = None, speed: float = 1.0):
        nn.Module.__init__(self)
        self.hamiltonian = hamiltonian
        self.speed = speed

        if T is None:
            self.input = nn.Parameter(randn())
        else:
            self.input = T

        self.geometry_like(self.hamiltonian)
        self.compile()

    def apply_gate_state(self, gate_state: GateState, psi: State):
        """use the geometry of the hamiltonian"""
        return self.hamiltonian.apply_gate_state(gate_state, psi)

    def compile(self):
        if not self.diag:
            # eigs, U = np.linalg.eigh(self.hamiltonian.H)
            eigs, U = torch._linalg_eigh(self.hamiltonian.H)
            self.eigs = torchcomplex(eigs)
            self.U = torchcomplex(U)

    def transform_eigs_H(self, fn):
        if self.diag:
            return fn(self.hamiltonian.H)
        else:
            return self.U @ (fn(self.eigs)[:, None] * self.U.T)

    def control(self, t: Scalar):
        return self.transform_eigs_H(lambda eigs: torch.exp(-1j * t.to(device) * eigs))

    # def dgate_state(self) -> GateState:
    #    if self.diag:
    #        dU = -1j * self.speed * self.hamiltonian.H * self.scaled_control()
    #    else:
    #        D = torch.exp(
    #            -1j * self.input * self.speed * self.eigs
    #            #-1j * self.input * self.speed * self.hamiltonian.strength * self.eigs
    #        )
    #        gate_state = self.U @ (D[:, None] * self.U.T)
    #        return gate_state

    #    return dU
