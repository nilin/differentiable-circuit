from differentiable_gate import Gate, State, Measurement, ThetaGate
from typing import Callable, List, Iterable, Optional, Union
from dataclasses import dataclass
import torch
from copy import deepcopy
from torch import nn
from collections import deque
from datatypes import *


class CircuitChannel(torch.nn.Module):
    gates: nn.ModuleList

    def apply(self, psi: State, randomness: Iterable[uniform01] = [], register=False):
        outcomes = []
        p_conditional = []
        checkpoints = []
        randomness = deque(randomness)
        for gate, where in self.flatgates_and_where():
            if not isinstance(gate, Measurement):
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

        for gate, where in self.flatgates_and_where()[::-1]:
            if isinstance(gate, ThetaGate):
                psi = gate.apply_reverse(psi)

                dU = gate.dgate_state()
                dE_input = 2 * cdot(X, gate.apply_gate_state(dU, psi)).real
                X = gate.apply_reverse(X)

                dE_inputs_rev.append(dE_input)
                inputs_rev.append(gate.input)

            elif isinstance(gate, Measurement):
                m = outcomes.pop()
                psi = checkpoints.pop()
                p = p_conditional.pop()
                X = gate.apply_reverse(X, m) / torch.sqrt(p)

            else:
                psi = gate.apply_reverse(psi)
                X = gate.apply_reverse(X)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X

    def flatgates_and_where(self) -> List[Tuple[Gate, Any]]:
        gates_and_where = []
        for component in self.gates:
            if isinstance(component, CircuitChannel):
                gates_and_where += [
                    (gate, (component,) + w) for gate, w in component.flatgates_and_where()
                ]
            else:
                gates_and_where.append((component, (component,)))
        return gates_and_where

    def reverse(self):
        self.gates = nn.ModuleList([gate.reverse() for gate in self.gates[::-1]])
        return self

    def get_reverse(self):
        self = deepcopy(self)
        return self.reverse()

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho):
        for i, (gate, where) in enumerate(self.flatgates_and_where()):
            rho = gate.apply_to_density_matrix(rho)
        return rho
