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

    def loss_and_grad_(
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

        x, dx = self.tangent(y, dy, reverse=True)
        return self.Lossgrad(loss, dx)
