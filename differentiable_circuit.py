from differentiable_gate import Gate, State, Measurement
from typing import Callable, List, Iterable
from dataclasses import dataclass
from gate_implementation import EvolveDensityMatrix
import torch
import numpy as np
from datatypes import *


def cdot(phi, psi):
    return phi.conj().dot(psi)


def squared_overlap(phi, psi):
    return torch.abs(cdot(phi, psi)) ** 2


class Params:
    def def_param(self, *initial_values):
        return (torch.tensor(v).requires_grad_() for v in initial_values)


@dataclass
class Circuit:
    gates: List[Gate]

    def apply(self, psi: State):
        for gate in self.gates:
            psi = gate.apply(psi)
        return psi

    def apply_to_density_matrix(self, rho):
        """for testing"""

        dm_impl = EvolveDensityMatrix()
        for gate in self.gates:
            rho = gate.apply(rho, implementation=dm_impl)
        return rho

    def optimal_control(self, psi: State, Obs: Callable[[State], State]):
        psi_t = self.apply(psi)
        Xt = Obs(psi_t)
        E = Xt.conj().dot(psi_t).real

        return E, self.backprop(psi_t, Xt)

    def backprop(self, psi, X):
        dE_inputs_rev = []
        inputs_rev = []

        for gate in self.gates[::-1]:
            psi_past = gate.reverse(psi)

            d_Uinv = gate.dgate_state(reverse=True)
            dE_input = 2 * cdot(psi_past, gate.apply_gate_state(d_Uinv, X)).real
            psi = psi_past
            X = gate.reverse(X)

            dE_inputs_rev.append(dE_input)
            inputs_rev.append(gate.input)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X


uniform01 = float


@dataclass
class Channel(Circuit):
    blocks: List[Circuit]
    measurements: List[Measurement]

    def apply(self, psi: State, randomness: Iterable[uniform01], register: bool = False):
        outcomes = []
        p_conditional = []
        checkpoints = []
        for block, M, u in zip(self.blocks, self.measurements, randomness):
            psi = block.apply(psi)
            if register:
                checkpoints.append(psi)

            psi, m, p = M.apply(psi, u)
            outcomes.append(m)
            p_conditional.append(p)

        if register:
            return psi, outcomes, p_conditional, checkpoints
        else:
            return psi

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[[State], State],
        randomness: Iterable[uniform01],
    ):
        psi_t, o, p, ch = self.apply(psi, randomness, register=True)
        Xt = Obs(psi_t)
        E = Xt.conj().dot(psi_t).real

        return E, self.backprop(psi_t, Xt, o, p, ch)

    def backprop(self, psi, X, outcomes, p_conditional, checkpoints):
        ps = torch.cumprod(torch.stack(p_conditional), 0)

        for block, M, p in reversed(list(zip(self.blocks, self.measurements, ps))):
            m = outcomes.pop()
            psi = checkpoints.pop()
            X = M.reverse(X, m)

            X = block.backprop(psi, X / p)
        return X

    def apply_to_density_matrix(self, rho):
        """for testing"""

        for block, M in zip(self.blocks, self.measurements):
            rho = block.apply(rho, implementation=EvolveDensityMatrix())
            rho = M.apply_to_density_matrix(rho)
        return rho
