from differentiable_gate import Gate, State, Measurement, uniform01
from typing import Callable, List
from dataclasses import dataclass
import torch


def overlap(phi, psi):
    return phi.conj().dot(psi)


class Params(dict):
    @staticmethod
    def def_param(*initial_values):
        return (torch.tensor(v).requires_grad_() for v in initial_values)


"""Circuit is less general than Channel below, but we first define the unitary version for readability"""


@dataclass
class Circuit:
    gates: List[Gate]

    def apply(self, x: State):
        for gate in self.gates:
            x = gate.apply(x)
        return x

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[[State], State],
    ):
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
            dE_input = 2 * overlap(psi_past, gate.apply_gate_state(d_Uinv, X)).real
            psi = psi_past
            X = gate.reverse(X)

            dE_inputs_rev.append(dE_input)
            inputs_rev.append(gate.input)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X


"""Channel generalizes Circuit to allow for measurements"""


@dataclass
class Channel(Circuit):
    gates: List[Gate]

    def apply(self, x: State, randomness: List[uniform01]):
        outcomes = []
        for gate in self.gates:
            if isinstance(gate, Measurement):
                x, m = gate.apply(x, u=randomness.pop())
                outcomes.append(m)
            else:
                x = gate.apply(x)

        return x, outcomes

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[[State], State],
        randomness,
    ):
        psi_t, outcomes = self.apply(psi, randomness)
        Xt = Obs(psi_t)
        E = Xt.conj().dot(psi_t).real

        return E, self.backprop(psi_t, Xt, outcomes)

    def backprop(self, psi, X, outcomes):
        dE_inputs_rev = []
        inputs_rev = []

        for gate in self.gates[::-1]:
            if isinstance(gate, Measurement):
                m = outcomes.pop()
                psi = gate.reverse(psi, m)
                X = gate.reverse(X, m)

            else:
                psi_past = gate.reverse(psi)

                d_Uinv = gate.dgate_state(reverse=True)
                dE_input = 2 * overlap(psi_past, gate.apply_gate_state(d_Uinv, X)).real
                psi = psi_past
                X = gate.reverse(X)

                dE_inputs_rev.append(dE_input)
                inputs_rev.append(gate.input)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X
