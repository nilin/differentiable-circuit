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

    def __init__(self, gates: List[Gate]):
        torch.nn.Module.__init__(self)
        self.gates = nn.ModuleList(gates)

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
        Obs: Callable[State, State] = None,
        outcome_values: List[float] = None,
        randomness: Iterable[uniform01] = [],
    ):
        psi_t, o, p, ch = self.apply(psi, randomness, register=True)

        E = 0

        if Obs is None:
            Xt = torch.zeros_like(psi_t)
        else:
            E += Obs(psi_t)
            (Xt,) = torch.autograd.grad(E, psi_t, retain_graph=True)
            Xt = Xt.conj()

        if outcome_values is None:
            dVal_dp = [0.0] * len(o)
        else:
            E += sum([o_i * v_i for o_i, v_i in zip(o, outcome_values)])
            dVal_dp = outcome_values

        return E, p, self.backprop(psi_t, Xt, dVal_dp, o, p, ch)

    def backprop(self, psi, X, dObs_dp, outcomes, p_conditional, checkpoints):
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

            elif isinstance(gate, Measurement):
                psi = checkpoints.pop()
                m = outcomes.pop()
                p = p_conditional.pop()
                X = gate.apply_reverse(X, m) / torch.sqrt(p)

                dvdp = dObs_dp.pop()
                X += dvdp * 2 * psi.conj()

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

    def _reverse(self, seen, **kwargs):
        newgates = []
        for gate in self.gates[::-1]:
            if gate in seen:
                newgates.append(gate)
            else:
                seen.add(gate)
                newgates.append(gate._reverse(seen=seen, **kwargs))

        self.gates = nn.ModuleList(newgates)
        return self

    def get_reverse(self, **kwargs):
        self = deepcopy(self)
        return self._reverse(seen=set(), **kwargs)

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho):
        for i, (gate, where) in enumerate(self.flatgates_and_where()):
            rho = gate.apply_to_density_matrix(rho)
        return rho
