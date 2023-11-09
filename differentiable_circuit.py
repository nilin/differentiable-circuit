from typing import Callable, List, Tuple
import torch
from collections import namedtuple
from differentiable_gate import Gate, State
from differentiable_gate import Scalar
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

    def tangent(self, x: State, dx: State, reverse=False):
        gates = self.gates[::-1] if reverse else self.gates

        for gate in gates:
            x, dx, dtheta = gate.tangent(x, dx, inverse=reverse)
            torch.autograd.backward(gate.input, dtheta)

        return x, dx

    def loss_and_grad(
        self,
        x: State,
        lossfn: Callable[[State], Scalar] = None,
        lossfn_and_grad: Callable[[State], Tuple[Scalar, State]] = None,
    ):
        y = self.apply(x)

        if lossfn_and_grad is not None:
            loss, dy = lossfn_and_grad(y)

        else:
            y.requires_grad_(True)
            loss = lossfn(y)
            loss.backward()
            dy = y.grad
            y.zero_grad()

        x, dx = self.tangent(y, dy, reverse=True)
        return self.Lossgrad(loss, dx)
