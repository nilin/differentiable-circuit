from . import config
from typing import Optional
from dataclasses import dataclass, KW_ONLY
import torch
from . import gate_implementation
from torch.autograd.functional import jacobian as torch_jacobian
from .datatypes import *
from collections import namedtuple
from torch import nn
import warnings
from functools import partial


from .datatypes import GateState, State, uniform01


class Op:
    def apply(self, psi: State):
        raise NotImplementedError

    def create_dense_matrix(self, n):
        I = torch.eye(2**n, dtype=tcomplex, device=config.device)
        return self.apply(I)

    def apply_to_density_matrix(self, rho: DensityMatrix):
        M_rho = self.apply(rho)
        M_rho_Mt = self.apply(M_rho.T.conj())
        return M_rho_Mt


class Gate(Op):
    """
    Classes inheriting from Gate need to specify the following:
    diag: bool
    k: int
    """

    ignored_positions: Tuple[int] = ()
    k: int
    diag: bool
    positions: Tuple[int]

    def __init__(self, positions):
        assert len(positions) == self.k
        self.positions = positions

    def apply_gate_state(self, gate_state: GateState, psi: State):
        if self.diag:
            gate = partial(gate_implementation.apply_gate_diag, self.positions, gate_state)
            # return gate_implementation.apply_gate_diag(self.positions, gate_state, psi)
        else:
            gate = partial(gate_implementation.apply_gate, self.positions, gate_state)
            # return gate_implementation.apply_gate(self.positions, gate_state, psi)

        # return gate(psi)
        return gate_implementation.apply_on_complement(self.ignored_positions, gate, psi)

    def adjoint(self, gate_state: GateState) -> GateState:
        if self.diag:
            return gate_state.conj()
        else:
            return gate_state.conj().T

    def geometry_like(self, gate: "Gate"):
        self.diag = gate.diag
        self.k = gate.k
        self.positions = gate.positions

    """
    Test utilities.
    """

    def apply_reverse(self, psi: State):
        raise NotImplementedError

    def _reverse(self, **kwargs):
        raise NotImplementedError

    def set_ignored_positions(self, ignored_positions):
        self.ignored_positions = ignored_positions
        return self


class ThetaGate(Gate):
    positions: Tuple[int]
    input: Scalar
    speed: float

    def __init__(self, positions, input: Scalar = None):
        super().__init__(positions)
        self.input = input
        self.speed = 1.0

    def apply(self, psi: State, **kwargs):
        gate_state=self.control(self.speed * self.input)
        return self.apply_gate_state(gate_state, psi, **kwargs)

    def dgate_state(self) -> GateState:
        """
        To be implemented by subclasses (e.g. Exp_iH).
        (e^-itH)'=-iH e^-itH
        """
        raise NotImplementedError

    def apply_reverse(self, psi: State):
        gate_state=self.control(-self.speed * self.input)
        return self.apply_gate_state(gate_state, psi)

    def control(self, theta: Scalar) -> GateState:
        """
        GateStates are small tensors that represent the local operation of a gate.
        We use autograd for the mapping from inputs (real parameter) to the GateState.

        The declaration of control depends on the specific gate type
        (X rotation, etc.).
        """
        raise NotImplementedError

    def set_direction_forward(self):
        self.speed = abs(self.speed)
        return self

    def set_direction_backward(self):
        self.speed = -abs(self.speed)
        return self


class DenseGate(Gate):
    k = None
    positions = ()

    def apply_gate_state(self, gate_state: GateState, psi: State):
        if self.diag:
            gate = lambda psi: gate_state.to(tcomplex) * psi
        else:
            gate = lambda psi: gate_state.to(tcomplex) @ psi

        return gate_implementation.apply_on_complement(self.ignored_positions, gate, psi)
