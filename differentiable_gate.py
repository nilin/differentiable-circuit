from typing import Callable, Tuple
from collections import namedtuple
from dataclasses import dataclass, field, KW_ONLY
import torch
import config

from torch.autograd.functional import jacobian as torch_jacobian


State = torch.Tensor
GateState = torch.Tensor
Scalar = torch.Tensor
ControlParams = Tuple[torch.Tensor]


class GateImplementation:
    apply_gate_1q: Callable[[GateState, State], State]
    apply_gate_2q: Callable[[GateState, State], State]
    apply_gate_diag: Callable[[GateState, State], State]


@dataclass
class Gate:
    Tangent = namedtuple("Tangent", ["y", "dy", "dtheta"])

    input: Scalar

    _: KW_ONLY
    gate_implementation: GateImplementation = field(
        default_factory=config.get_default_gate_implementation
    )

    def apply_gate_state(self, gate_state: torch.Tensor, x: State) -> State:
        """
        The declaration of apply depends on the gate type (geometry, etc.),
        not on the implementation
        """
        raise NotImplementedError

    def apply(self, x: State, inverse=False):
        gate_state = self.control(self.input)
        if inverse:
            gate_state = self.adjoint(gate_state)
        return self.apply_gate_state(gate_state, x)

    def dgate_state(self, inverse=False) -> GateState:
        dU = self.complex_out_jacobian(self.control, self.input)
        if inverse:
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

    @staticmethod
    def adjoint(gate_state: GateState) -> GateState:
        inv = gate_state.conj().T
        return inv

    @staticmethod
    def complex_out_jacobian(f, t):
        real = torch_jacobian(lambda x: f(x).real, t)
        imag = torch_jacobian(lambda x: f(x).imag, t)
        return torch.complex(real, imag)


@dataclass
class Gate_1q(Gate):
    p: int

    def apply_gate_state(self, gate_state, x):
        return self.gate_implementation.apply_gate_1q(gate_state, x, self.p)


@dataclass
class Gate_2q(Gate):
    p: int
    q: int

    def apply_gate_state(self, gate_state, x):
        return self.gate_implementation.apply_gate_2q(
            gate_state, x, self.p, self.q
        )


class Diag(Gate):
    def adjoint(self, gate):
        return gate.conj()


class Gate_2q_diag(Gate_2q, Diag):
    def apply_gate_state(self, gate_state, x):
        return self.gate_implementation.apply_gate_2q_diag(
            gate_state, x, self.p, self.q
        )
