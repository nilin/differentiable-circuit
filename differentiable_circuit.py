from typing import Callable, List
import torch
from collections import namedtuple
from differentiable_gate import Gate, State
from dataclasses import dataclass


@dataclass
class UnitaryCircuit:
    Lossgrad = namedtuple("Lossgrad", ["loss", "dx"])
    gates: List[Gate]

    def apply(self, x: State, reverse=False):
        gates = self.gates[::-1] if reverse else self.gates

        for gate in gates:
            x = gate.apply(x, inverse=reverse)
        return x

    def tangent(self, psi: State, X: State, reverse=False):
        gates = self.gates[::-1] if reverse else self.gates

        for gate in gates:
            psi, X, dtheta = gate.tangent(psi, X, inverse=reverse)
            torch.autograd.backward(gate.input, 2 * dtheta.real)

        return psi, X

    def loss_and_grad(
        self,
        psi: State,
        Op: Callable[[State], State] = None,
    ):
        psi_t = self.apply(psi)
        Xt = Op(psi_t)
        loss = Xt.conj().dot(psi_t).real

        psi0, X0 = self.tangent(psi_t, Xt, reverse=True)
        return loss, X0


class Params(dict):
    # def __init__(self):
    #    super().__init__()

    # __getattr__ = dict.get
    # __setattr__ = dict.__setitem__
    # __delattr__ = dict.__delitem__

    # def add(self, **kwargs):
    #    for key, value in kwargs.items():
    #        self.__setitem__(key, self.def_param(value))
    #    return self.values()

    @staticmethod
    def def_param(*initial_values):
        return [torch.tensor(v).requires_grad_() for v in initial_values]
