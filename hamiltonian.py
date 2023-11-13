from differentiable_gate import Scalar, Gate
from differentiable_circuit import Circuit, cdot, State, Channel
from typing import Callable, List
import copy
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

    def control(self, t: ignore):
        return self.strength * self.H


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

    @staticmethod
    def TrotterSuzuki(
        Layer1: List[HamiltonianTerm],
        Layer2: List[HamiltonianTerm],
        T: Scalar,
        steps: int,
    ):
        U_0 = [Exp_i(H, T, 1 / (2 * steps)) for H in Layer2]
        U_1 = [Exp_i(H, T) for H in Layer1]
        U_2 = [Exp_i(H, T) for H in Layer2]
        return U_0 + (U_1 + U_2) * (steps - 1) + U_1 + U_0


class Exp_i(Gate):
    def __init__(self, hamiltonian: HamiltonianTerm, T: Scalar, speed: float = 1.0):
        self.hamiltonian = hamiltonian
        self.input = T
        self.speed = speed

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
