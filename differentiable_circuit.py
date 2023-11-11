from differentiable_gate import Gate, State
from typing import Callable, List, Iterable
from dataclasses import dataclass
from gate_implementation import torchcomplex
import torch
import numpy as np


def cdot(phi, psi):
    return phi.conj().dot(psi)


def squared_overlap(phi, psi):
    return torch.abs(cdot(phi, psi)) ** 2


class Params:
    def def_param(self, *initial_values):
        return (torch.tensor(v).requires_grad_() for v in initial_values)


"""Circuit is less general than Channel below, but we first define the unitary version for readability"""


@dataclass
class Circuit:
    gates: List[Gate]

    def apply(self, x: State):
        for gate in self.gates:
            x = gate.apply(x)
        return x

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


"""Channel generalizes Circuit to allow for measurements"""

uniform01 = float


@dataclass
class Channel(Circuit):
    blocks: List[Circuit]
    measurements: List[Gate]

    def apply(self, x: State, randomness: Iterable[uniform01], register: bool = False):
        outcomes = []
        p_conditional = []
        checkpoints = []
        for block, M, u in zip(self.blocks, self.measurements, randomness):
            x = block.apply(x)
            if register:
                checkpoints.append(x)

            x, m, p = M.apply(x, u)
            outcomes.append(m)
            p_conditional.append(p)

        if register:
            return x, outcomes, p_conditional, checkpoints
        else:
            return x

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
