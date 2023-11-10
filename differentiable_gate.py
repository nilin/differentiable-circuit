from typing import Callable, Tuple, Optional
import torch
import config
from dataclasses import dataclass, field

from torch.autograd.functional import jacobian as torch_jacobian


State = torch.Tensor
GateState = torch.Tensor
Scalar = torch.Tensor
ControlParams = Tuple[torch.Tensor]
uniform01 = float


class GateImplementation:
    apply_gate_1q: Callable[[GateState, State], State]
    apply_gate_2q: Callable[[GateState, State], State]
    apply_gate_diag: Callable[[GateState, State], State]


@dataclass(kw_only=True)
class Gate:
    input: Optional[Scalar] = None
    gate_implementation: GateImplementation = field(
        default_factory=config.get_default_gate_implementation
    )

    def apply_gate_state(self, gate_state: torch.Tensor, x: State) -> State:
        """
        The declaration of apply depends on the gate type (geometry, etc.),
        not on the implementation
        """
        raise NotImplementedError

    def apply(self, x: State):
        gate_state = self.control(self.input)
        return self.apply_gate_state(gate_state, x)

    def reverse(self, x: State):
        gate_state = self.control(self.input)
        gate_state = self.adjoint(gate_state)
        return self.apply_gate_state(gate_state, x)

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


"""Types of gate geometry"""


@dataclass
class Gate_2q(Gate):
    p: int
    q: int

    def apply_gate_state(self, gate_state, x):
        return self.gate_implementation.apply_gate_2q(gate_state, x, self.p, self.q)


@dataclass
class Diag(Gate):
    def adjoint(self, gate):
        return gate.conj()


@dataclass
class Gate_2q_diag(Gate_2q, Diag):
    def apply_gate_state(self, gate_state, x):
        return self.gate_implementation.apply_gate_2q_diag(gate_state, x, self.p, self.q)


"""Non-unitary gates"""


class Measurement(Gate):
    def apply(self, x: State, u: uniform01):
        N = len(x)
        p0 = probabilitymass(x[: N // 2]) / probabilitymass(x)

        outcome = u > p0
        return self.cut(x, outcome), outcome

    def cut(self, x: State, outcome: bool):
        N = len(x)
        if outcome:
            return x[N // 2 :]
        else:
            return x[: N // 2]

    def embed(self, x: State, outcome: bool):
        if outcome:
            return torch.cat((torch.zeros_like(x), x))
        else:
            return torch.cat((x, torch.zeros_like(x)))

    def reverse(self, x: State, outcome: bool):
        return self.embed(x, outcome)


def probabilitymass(x):
    return torch.sum(torch.abs(x) ** 2).real


class CleanSlateAncilla(Measurement):
    def apply(self, x: State, u: uniform01):
        N = len(x)
        p0 = probabilitymass(x[: N // 2]) / probabilitymass(x)

        outcome = u > p0
        x = self.embed(self.cut(x, outcome), 0)
        return x, outcome

    def reverse(self, x: State, outcome: bool):
        return self.embed(self.cut(x, 0), outcome)
