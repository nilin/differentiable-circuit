import config
from typing import Callable, Tuple, Optional, Any
from dataclasses import dataclass, field, KW_ONLY
import torch
from torch.autograd.functional import jacobian as torch_jacobian
from datatypes import *
from collections import namedtuple


@dataclass
class Gate:
    implementation = config.get_default_gate_implementation()
    """
    Classes inheriting from Gate need to specify the following:
    diag: bool
    k: int
    """

    p: Optional[int] = None
    q: Optional[int] = None
    _: KW_ONLY
    input: Optional[Scalar] = None

    def apply(self, psi: State, **kwargs):
        gate_state = self.control(self.input)
        return self.apply_gate_state(gate_state, psi, **kwargs)

    def apply_gate_state(
        self,
        gate_state: GateState,
        psi: State,
        implementation: GateImplementation = None,
    ):
        if implementation is None:
            return self.implementation.apply_gate(self, gate_state, psi)
        else:
            return implementation.apply_gate(self, gate_state, psi)

    def reverse(self, psi: State):
        gate_state = self.control(self.input)
        gate_state = self.adjoint(gate_state)
        return self.apply_gate_state(gate_state, psi)

    def dgate_state(self, reverse) -> GateState:
        dU = self.complex_out_jacobian(self.control, self.input)
        if reverse:
            return self.adjoint(dU)
        else:
            return dU

    def control(self, theta: Scalar) -> GateState:
        """
        The declaration of control depends on the specific gate type
        (X rotation, etc.). The implementation of control needs to use Tensor
        operations throughout to make use of autograd.
        """
        raise NotImplementedError

    def set_input(self, input):
        self.input = input
        return self

    def adjoint(self, gate_state: GateState) -> GateState:
        if self.diag:
            return gate_state.conj()
        else:
            return gate_state.conj().T

    def geometry_like(self, gate: "Gate"):
        self.diag = gate.diag
        self.k = gate.k

        if self.k == 1:
            self.p = gate.p

        if self.k == 2:
            self.p = gate.p
            self.q = gate.q

    @staticmethod
    def complex_out_jacobian(f, t):
        real = torch_jacobian(lambda x: f(x).real, t)
        imag = torch_jacobian(lambda x: f(x).imag, t)
        return torch.complex(real, imag)


@dataclass
class Measurement(Gate):
    implementation = config.get_default_gate_implementation()
    outcome_tuple = namedtuple("Measurement", ["psi", "outcome", "p_outcome"])
    p: int

    def measure(self, psi: State, u: uniform01, normalize=True):
        _0, _1 = self.implementation.split_by_bit_p(len(psi), self.p)
        p0 = probabilitymass(psi[_0]) / probabilitymass(psi)
        outcome = u > p0
        p_outcome = p0 if outcome == 0 else 1 - p0
        indices = [_0, _1][outcome]

        if normalize:
            psi_post = psi[indices] / torch.sqrt(p_outcome)
        else:
            psi_post = psi[indices]

        return self.outcome_tuple(psi_post, outcome, p_outcome)

    def partial_trace(self, rho: DensityMatrix):
        """used for testing"""

        _0, _1 = self.implementation.split_by_bit_p(len(rho), self.p)
        return rho[_0, _0] + rho[_1, _1]


class CleanSlateAncilla(Measurement):
    def apply(self, psi: State, u: uniform01, **kwargs):
        psi_post, outcome, p_outcome = self.measure(psi, u, **kwargs)

        psi_out = torch.zeros_like(psi)
        _0, _1 = self.implementation.split_by_bit_p(len(psi), self.p)
        psi_out[_0] = psi_post
        return self.outcome_tuple(psi_out, outcome, p_outcome)

    def reverse(self, psi: State, outcome: bool):
        _0, _1 = self.implementation.split_by_bit_p(len(psi), self.p)
        psi_out = torch.zeros_like(psi)
        psi_out[[_0, _1][outcome]] = psi[_0]
        return psi_out

    def apply_to_density_matrix(self, rho: State):
        self.partial_trace(rho)
        rho_out = torch.zeros_like(rho)
        _0, _1 = self.implementation.split_by_bit_p(len(rho), self.p)
        rho_out[_0, _0] = self.partial_trace(rho)
        return rho_out


class Identity(Gate):
    def apply(self, x: State, **kwargs):
        return x

    def reverse(self, x: State):
        return x
