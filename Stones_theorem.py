from differentiable_gate import Scalar, Gate, CleanSlateAncilla
from differentiable_circuit import Circuit, Params, overlap, State, Channel
from typing import Callable, List
import copy
from dataclasses import dataclass
import torch
import torch
from differentiable_gate import (
    Gate,
    Diag,
    State,
)
import numpy as np
from gate_implementations import torchcomplex


"""Define gates as Hamiltonian evolution"""


@dataclass(kw_only=True)
class Exp_iH(Gate):
    strength: float = 1.0

    def __post_init__(self):
        self.compile()

    def compile(self):
        eigs, U = np.linalg.eigh(self.H)
        self.eigs = torchcomplex(eigs)
        self.U = torchcomplex(U)

    def control(self, t):
        D = torch.exp(-1j * t * self.strength * self.eigs)
        gate_state = self.U @ (D[:, None] * self.U.T)
        return gate_state

    def as_observable(self, psi: State):
        self.apply_gate_state(self.H, psi)


class Exp_iH_diag(Exp_iH, Diag):
    def compile(self):
        pass

    def control(self, t):
        return torch.exp(-1j * t * self.strength * self.H)


"""Define a Hamiltonian as a sum of local terms (named according to their unitary evolution)"""


@dataclass
class Hamiltonian:
    terms = List[Exp_iH]

    def apply_H(self, psi: State):
        H_psi = torch.zeros_like(psi)
        for U_i in self.terms:
            H_psi += U_i.as_observable(psi) * U_i.strength
        return H_psi

    def expectation(self, psi: State):
        return psi.conj().dot(self.apply_H(psi))

    @staticmethod
    def TrotterSuzuki(
        Layer1: List[Exp_iH],
        Layer2: List[Exp_iH],
        T: Scalar,
        steps: int,
    ):
        t = T / steps
        U_0 = [copy.deepcopy(U).set_input(t / 2) for U in Layer2]
        U_1 = [U.set_input(t) for U in Layer1]
        U_2 = [U.set_input(t) for U in Layer2]
        return U_0 + (U_1 + U_2) * (steps - 1) + U_1 + U_0
