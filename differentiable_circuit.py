from typing import Callable, List
import torch
from differentiable_gate import Gate, State
from dataclasses import dataclass


def overlap(phi, psi):
    return phi.conj().dot(psi)


class Params(dict):
    @staticmethod
    def def_param(*initial_values):
        return (torch.tensor(v).requires_grad_() for v in initial_values)


@dataclass
class UnitaryCircuit:
    gates: List[Gate]

    def apply(self, x: State, inverse=False):
        gates = self.gates[::-1] if inverse else self.gates

        for gate in gates:
            x = gate.apply(x, inverse=inverse)
        return x

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[[State], State] = None,
    ):
        psi_t = self.apply(psi)
        Xt = Obs(psi_t)
        E = Xt.conj().dot(psi_t).real

        return E, self.backprop(psi_t, Xt)

    def backprop(self, psi, X):
        dE_inputs_rev=[]
        inputs_rev=[]

        for gate in self.gates[::-1]:
            psi_past = gate.apply(psi, inverse=True)

            d_Uinv = gate.dgate_state(inverse=True)
            dE_input = (
                2 * overlap(psi_past, gate.apply_gate_state(d_Uinv, X)).real
            )
            psi = psi_past
            X = gate.apply(X, inverse=True)

            dE_inputs_rev.append(dE_input)
            inputs_rev.append(gate.input)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X
