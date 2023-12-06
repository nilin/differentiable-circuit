from differentiable_gate import Gate, State, ThetaGate
from non_unitary_gates import Measurement
from typing import Callable, List, Iterable, Optional, Union
from dataclasses import dataclass
import torch
from copy import deepcopy
from torch import nn
from collections import deque
from datatypes import *


class Circuit(torch.nn.Module):
    direction_forward = True
    gates: nn.ModuleList

    def __init__(self, gates: List[Gate]):
        torch.nn.Module.__init__(self)
        self.gates = nn.ModuleList(gates)

    def apply(self, psi: State):
        for gate, where in self.flatgates_and_where():
            psi = gate.apply(psi)
        return psi

    """
    for nested constructions and tracing current position
    """

    def flatgates_and_where(self) -> List[Tuple[Gate, Any]]:
        gates_and_where = []

        gates = self.gates if self.direction_forward else self.gates[::-1]

        for component in gates:
            if isinstance(component, Circuit):
                gates_and_where += [
                    (gate, (component,) + w) for gate, w in component.flatgates_and_where()
                ]
            else:
                gates_and_where.append((component, (component,)))

        return gates_and_where

    """
    for applying circuit in reverse
    """

    def set_direction_forward(self):
        for gate in self.gates:
            gate.set_direction_forward()
        self.direction_forward = True
        return self

    def set_direction_backward(self):
        for gate in self.gates:
            gate.set_direction_backward()
        self.direction_forward = False
        return self

    def do_backward(self, fn, *args, **kwargs):
        assert self.direction_forward
        self.set_direction_backward()
        out = fn(*args, **kwargs)
        self.set_direction_forward()
        return out

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho, detach=False):
        for i, (gate, where) in enumerate(self.flatgates_and_where()):
            rho = gate.apply_to_density_matrix(rho)
            if detach:
                rho = rho.detach()
        return rho


class UnitaryCircuit(Circuit, torch.autograd.Function):
    def optimal_control(
        self,
        psi: State,
        Obs: Callable[State, Scalar] = None,
        target: State = None,
    ):
        psi_t = self.apply(psi)

        if target is None:
            E = Obs(psi_t)
            (Xt,) = torch.autograd.grad(E, psi_t, retain_graph=True)
            Xt = Xt.conj()

        else:
            E = squared_overlap(target, psi_t)
            Xt = cdot(target, psi_t) * target

        return E, self.backprop(psi_t, Xt)

    def backprop(self, psi, X):
        dE_inputs_rev = []
        inputs_rev = []

        for gate, where in self.flatgates_and_where()[::-1]:
            if isinstance(gate, ThetaGate):
                psi = gate.apply_reverse(psi)

                dU = gate.dgate_state()
                dE_input = cdot(X, gate.apply_gate_state(dU, psi)).real
                X = gate.apply_reverse(X)

                dE_inputs_rev.append(dE_input)
                inputs_rev.append(gate.input)

            else:
                psi = gate.apply_reverse(psi)
                X = gate.apply_reverse(X)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X

    def forward(self, psi: State):
        for gate, where in self.flatgates_and_where():
            psi = gate.apply(psi)
        return psi

    def apply(self, psi: State):
        return self.forward(psi)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(outputs)

    def backward(self, ctx, grad_output):
        psi = ctx.saved_tensors[0]
        X = grad_output.conj()
        return self.backprop(psi, X)


class SquaredOverlap:
    def __init__(self, target: State):
        self.target = target

    def __call__(self, psi: State):
        return squared_overlap(self.target, psi)

    def forward(self, psi: State):
        return squared_overlap(self.target, psi)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(inputs)

    def backward(self, ctx, grad_output):
        return cdot(self.target, ctx.saved_tensors[0]) * self.target


class Non_unitary_circuit(Circuit):
    def apply_and_register(self, psi: State):
        outcomes = []
        p_conditional = []
        checkpoints = []
        randomness = deque(self.make_randomness())

        for gate, where in self.flatgates_and_where():
            if not isinstance(gate, Measurement):
                psi = gate.apply(psi)
            else:
                checkpoints.append(psi)
                u = randomness.popleft()
                psi, m, p = gate.measure(psi, u=u)
                outcomes.append(m)
                p_conditional.append(p.cpu())

        return psi, outcomes, p_conditional, checkpoints

    def make_randomness(self):
        nmeasurements = len(
            [
                gate
                for gate, where in self.flatgates_and_where()
                if isinstance(gate, Measurement)
            ]
        )
        return torch.rand(nmeasurements)
