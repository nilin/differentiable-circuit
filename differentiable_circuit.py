from differentiable_gate import Gate, State, Measurement
from typing import Callable, List, Iterable
from dataclasses import dataclass
from gate_implementation import EvolveDensityMatrix
import torch
from collections import deque
import numpy as np
from datatypes import *


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
        expectation = Xt.conj().dot(psi_t).real

        return expectation, self.backprop(psi_t, Xt)

    def backprop(self, psi, X):
        dE_inputs_rev = []
        inputs_rev = []

        for gate in self.gates[::-1]:
            psi = gate.reverse(psi)

            dU = gate.dgate_state()
            dE_input = 2 * cdot(X, gate.apply_gate_state(dU, psi)).real
            X = gate.reverse(X)

            dE_inputs_rev.append(dE_input)
            inputs_rev.append(gate.input)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X


class Channel(Circuit):
    def apply(self, psi: State, randomness: Iterable[uniform01], register=False):
        outcomes = []
        p_conditional = []
        checkpoints = []
        randomness = deque(randomness)
        for gate in self.gates:
            if gate.unitary:
                psi = gate.apply(psi)
            else:
                if register:
                    checkpoints.append(psi)

                u = randomness.popleft()
                psi, m, p = gate.apply(psi, u=u, normalize=True)
                outcomes.append(m)
                p_conditional.append(p.cpu())

        if register:
            return psi, outcomes, p_conditional, checkpoints
        else:
            return psi

    def apply_to_density_matrix(self, rho):
        """for testing"""

        dm_impl = EvolveDensityMatrix()
        for gate in self.gates:
            if gate.unitary:
                rho = gate.apply(rho, implementation=dm_impl)
            else:
                rho = gate.apply_to_density_matrix(rho)
        return rho

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[[State], State],
        randomness: Iterable[uniform01],
    ):
        psi_t, o, p, ch = self.apply(psi, randomness, register=True)
        Xt = Obs(psi_t)
        E = Xt.conj().dot(psi_t).real

        return E, self.backprop(psi_t, Xt, o, p, ch, E)

    def backprop(self, psi, X, outcomes, p_conditional, checkpoints, E):
        dE_inputs_rev = []
        inputs_rev = []

        for gate in self.gates[::-1]:
            if gate.unitary:
                psi = gate.reverse(psi)

                dU = gate.dgate_state()
                dE_input = 2 * cdot(X, gate.apply_gate_state(dU, psi)).real
                X = gate.reverse(X)

                dE_inputs_rev.append(dE_input)
                inputs_rev.append(gate.input)

            else:
                m = outcomes.pop()
                psi = checkpoints.pop()
                p = p_conditional.pop()
                X = X + E * psi / p
                X = gate.reverse(X, m)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X
