from differentiable_gate import Gate, State
from typing import Callable, List, Iterable
from dataclasses import dataclass
import torch
from collections import deque
from datatypes import *


@dataclass
class CircuitChannel:
    gates: List[Gate]

    def apply(self, psi: State, randomness: Iterable[uniform01] = [], register=False):
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

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[[State], State],
        randomness: Iterable[uniform01] = [],
    ):
        psi_t, o, p, ch = self.apply(psi, randomness, register=True)
        Xt = Obs(psi_t)
        E = Xt.conj().dot(psi_t).real

        return E, self.backprop(psi_t, Xt, o, p, ch)

    def backprop(self, psi, X, outcomes, p_conditional, checkpoints):
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
                X = gate.reverse(X, m) / torch.sqrt(p)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho):
        for gate in self.gates:
            rho = gate.apply_to_density_matrix(rho)
        return rho
