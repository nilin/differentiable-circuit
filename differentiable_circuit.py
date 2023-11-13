from differentiable_gate import Gate, State, Measurement
from typing import Callable, List, Iterable
from dataclasses import dataclass
from gate_implementation import EvolveDensityMatrix
import torch
import numpy as np
from datatypes import *


@dataclass
class Circuit:
    gates: List[Gate]

    def apply(self, psi: State, randomness: ignore = None):
        for gate in self.gates:
            psi = gate.apply(psi)
        return psi

    def apply_to_density_matrix(self, rho):
        """for testing"""

        dm_impl = EvolveDensityMatrix()
        for gate in self.gates:
            rho = gate.apply(rho, implementation=dm_impl)
        return rho

    def optimal_control(
        self, psi: State, Obs: Callable[[State], State], randomness: ignore = None
    ):
        psi_t = self.apply(psi)
        Xt = Obs(psi_t)
        expectation = Xt.conj().dot(psi_t).real

        return expectation, self.backprop(psi_t, Xt)

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


@dataclass
class Channel:
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

            psi, m, p = M.apply(psi, u=u, normalize=True)
            outcomes.append(m)
            p_conditional.append(p.cpu())

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
        # ps = torch.cumprod(torch.stack([torch.tensor(1.0)] + p_conditional[:-1]), 0)
        # ps = torch.cumprod(torch.stack(p_conditional), 0)
        # for block, M, p in reversed(list(zip(self.blocks, self.measurements, ps))):
        #    m = outcomes.pop()
        #    psi = checkpoints.pop()
        #    X = M.reverse(X, m)
        #    # X = block.backprop(psi, X / p.to(X.device))
        #    X = block.backprop(psi, X)

        p = torch.prod(torch.stack(p_conditional), 0)
        for block, M in reversed(list(zip(self.blocks, self.measurements))):
            m = outcomes.pop()
            psi = checkpoints.pop()
            X = M.reverse(X, m)
            # X = block.backprop(psi, X / torch.sqrt(p.to(X.device)))
            X = block.backprop(psi, X)
        return X

    def apply_to_density_matrix(self, rho):
        """for testing"""

        for block, M in zip(self.blocks, self.measurements):
            rho = block.apply_to_density_matrix(rho)
            rho = M.apply_to_density_matrix(rho)
        return rho
