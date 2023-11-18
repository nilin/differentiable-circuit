from differentiable_gate import Scalar, Gate, ThetaGate
from differentiable_circuit import State, CircuitChannel
from typing import List
from torch import nn
import config
from dataclasses import dataclass
import torch
import torch
from differentiable_gate import (
    Gate,
    State,
)
import numpy as np
from datatypes import *


"""Define gates as Hamiltonian evolution"""


@dataclass(kw_only=True)
class HamiltonianTerm(Gate):
    """Classes inheriting from HamiltonianTerm need to specify H: GateState"""

    strength: float = 1.0

    def apply(self, psi: State):
        gate_state = self.strength * self.H
        return self.apply_gate_state(gate_state, psi)


@dataclass
class Hamiltonian:
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

    def create_dense(self, n):
        return Gate.create_dense(self, n)


class TrotterSuzuki(CircuitChannel):
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
            self.T = nn.Parameter(torch.randn(1, config.gen))
        else:
            self.T = T

        U_0 = [Exp_i(H, self.T, 1 / (2 * self.steps)) for H in self.Layer2]
        U_1 = [Exp_i(H, self.T) for H in self.Layer1]
        U_2 = [Exp_i(H, self.T) for H in self.Layer2]
        self.gates = U_0 + (U_1 + U_2) * (self.steps - 1) + U_1 + U_0


class Exp_i(ThetaGate, nn.Module):
    hamiltonian: HamiltonianTerm = None

    def __init__(self, hamiltonian: HamiltonianTerm, T: Scalar = None, speed: float = 1.0):
        self.hamiltonian = hamiltonian
        self.speed = speed

        nn.Module.__init__(self)
        if T is None:
            self.input = nn.Parameter(torch.randn(1, config.gen))
        else:
            self.input = T

        self.geometry_like(self.hamiltonian)
        self.compile()

    def compile(self):
        if not self.diag:
            eigs, U = np.linalg.eigh(self.hamiltonian.H)
            self.eigs = torchcomplex(eigs)
            self.U = torchcomplex(U)

    def control(self, t):
        if self.diag:
            return torch.exp(
                -1j * t * self.speed * self.hamiltonian.strength * self.hamiltonian.H
            )
        else:
            D = torch.exp(-1j * t * self.speed * self.hamiltonian.strength * self.eigs)
            gate_state = self.U @ (D[:, None] * self.U.T)
            return gate_state
