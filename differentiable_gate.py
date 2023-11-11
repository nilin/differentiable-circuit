import config
from typing import Callable, Tuple, Optional, Any
from dataclasses import dataclass, field, KW_ONLY
import torch
from torch.autograd.functional import jacobian as torch_jacobian


State = torch.Tensor
GateState = torch.Tensor
Scalar = torch.Tensor
ControlParams = Tuple[torch.Tensor]
uniform01 = float


class GateImplementation:
    apply_gate: Callable[[Any, GateState, State], State]


@dataclass
class Gate:
    """
    Classes inheriting from Gate need to specify the following:
    diag: bool
    k: int
    """

    p: Optional[int] = None
    q: Optional[int] = None
    _: KW_ONLY
    input: Optional[Scalar] = None
    implementation: GateImplementation = field(
        default_factory=config.get_default_gate_implementation
    )

    def apply(self, x: State):
        gate_state = self.control(self.input)
        return self.apply_gate_state(gate_state, x)

    def reverse(self, x: State):
        gate_state = self.control(self.input)
        gate_state = self.adjoint(gate_state)
        return self.apply_gate_state(gate_state, x)

    def apply_gate_state(self, gate_state: GateState, x: State):
        return self.implementation.apply_gate(self, gate_state, x)

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

    @staticmethod
    def complex_out_jacobian(f, t):
        real = torch_jacobian(lambda x: f(x).real, t)
        imag = torch_jacobian(lambda x: f(x).imag, t)
        return torch.complex(real, imag)
