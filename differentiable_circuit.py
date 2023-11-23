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
    forward = True
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
        for component in self.gates:
            if isinstance(component, Circuit):
                gates_and_where += [
                    (gate, (component,) + w) for gate, w in component.flatgates_and_where()
                ]
            else:
                gates_and_where.append((component, (component,)))

        if self.forward:
            return gates_and_where
        else:
            return gates_and_where[::-1]

    """
    for applying circuit in reverse
    """

    def set_direction_forward(self):
        for gate in self.gates:
            gate.set_direction_forward()
        self.forward = True
        return self

    def set_direction_backward(self):
        for gate in self.gates:
            gate.set_direction_backward()
        self.forward = False
        return self

    def do_backward(self, fn, *args, **kwargs):
        assert self.forward
        self.set_direction_backward()
        out = fn(*args, **kwargs)
        self.set_direction_forward()
        return out

    """
    Test utilities.
    """
    # def apply_to_density_matrix(self, rho, checkpoint_at=None):
    #    checkpoints = []

    #    for i, (gate, where) in enumerate(self.flatgates_and_where()):
    #        rho = gate.apply_to_density_matrix(rho)

    #        if checkpoint_at is not None and checkpoint_at(gate):
    #            checkpoints.append(rho)

    #    if checkpoint_at is not None:
    #        return rho, dict(checkpoints=checkpoints)
    #    else:
    #        return rho

    def apply_to_density_matrix(self, rho):
        for i, (gate, where) in enumerate(self.flatgates_and_where()):
            rho = gate.apply_to_density_matrix(rho)
        return rho


class UnitaryCircuit(Circuit):
    def optimal_control(
        self,
        psi: State,
        Obs: Callable[State, State] = None,
    ):
        psi_t = self.apply(psi)
        E = Obs(psi_t)
        (Xt,) = torch.autograd.grad(E, psi_t, retain_graph=True)
        Xt = Xt.conj()
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
