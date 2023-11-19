import config
from typing import Optional
from dataclasses import dataclass, KW_ONLY
import torch
import gate_implementation
from torch.autograd.functional import jacobian as torch_jacobian
from datatypes import *
from collections import namedtuple
from torch import nn

from datatypes import GateImplementation, GateState, State, uniform01


class Gate:
    implementation = config.get_default_gate_implementation()
    """
    Classes inheriting from Gate need to specify the following:
    diag: bool
    k: int
    """
    p: Optional[int]
    q: Optional[int]

    def __init__(self, p=None, q=None):
        self.p = p
        self.q = q

    def apply_gate_state(self, gate_state: GateState, psi: State):
        return self.implementation.apply_gate(self, gate_state, psi)

    def control(self, theta: Scalar) -> GateState:
        """
        GateStates are small tensors that represent the local operation of a gate.
        We use autograd for the mapping from inputs (real parameter) to the GateState.

        The declaration of control depends on the specific gate type
        (X rotation, etc.).
        """
        raise NotImplementedError

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

    """
    Test utilities.
    """

    def create_dense(self, n):
        I = torch.eye(2**n, dtype=tcomplex, device=config.device)
        return self.apply(I)

    def apply_to_density_matrix(self, rho: DensityMatrix):
        M_rho = self.apply(rho)
        M_rho_Mt = self.apply(M_rho.T.conj())
        return M_rho_Mt

    def apply_reverse(self, psi: State):
        raise NotImplementedError

    def _reverse(self, **kwargs):
        raise NotImplementedError


class ThetaGate(Gate):
    p: Optional[int]
    q: Optional[int]
    input: Scalar
    speed: float

    def __init__(self, p=None, q=None, input: Scalar = None):
        super().__init__(p=p, q=q)
        self.input = input
        self.speed = 1.0

    def apply(self, psi: State, **kwargs):
        gate_state = self.scaled_control(self.input)
        return self.apply_gate_state(gate_state, psi, **kwargs)

    def dgate_state(self) -> GateState:
        dU = self.complex_out_jacobian(self.scaled_control, self.input)
        return dU

    def apply_reverse(self, psi: State):
        gate_state = self.scaled_control(self.input)
        gate_state = self.adjoint(gate_state)
        return self.apply_gate_state(gate_state, psi)

    def scaled_control(self, theta: Scalar) -> GateState:
        return self.control(self.speed * theta)

    def _reverse(self, **kwargs):
        self.speed = -self.speed
        return self

    @staticmethod
    def complex_out_jacobian(f, t):
        real = torch_jacobian(lambda x: f(x).real, t)
        imag = torch_jacobian(lambda x: f(x).imag, t)
        return torch.complex(real, imag)


class AddAncilla(Gate, torch.nn.Module):
    p: int

    def __init__(self, p=0):
        torch.nn.Module.__init__(self)
        self.p = p

    def apply(self, psi: State):
        N = 2 * len(psi)
        psi_out = torch.zeros((N,) + psi.shape[1:], dtype=psi.dtype, device=psi.device)
        _0, _1 = self.implementation.split_by_bit_p(N, self.p)
        psi_out[_0] += psi
        return psi_out

    def apply_reverse(self, psi: State):
        return RestrictMeasurementOutcome.apply(self, psi)

    def _reverse(self, **kwargs):
        return RestrictMeasurementOutcome(self.p)


class RestrictMeasurementOutcome(AddAncilla):
    def apply(self, psi: State):
        _0, _1 = self.implementation.split_by_bit_p(len(psi), self.p)
        return psi[_0]

    def apply_reverse(self, psi: State):
        return AddAncilla.apply(self, psi)

    def _reverse(self, **kwargs):
        return AddAncilla(self.p)

    # def apply_to_density_matrix(self, rho: DensityMatrix):
    #    _0, _1 = gate_implementation.split_by_bit_p(len(rho), self.p)
    #    return rho[_0][:, _0]


class AddRandomAncilla(Gate, torch.nn.Module):
    p: int

    def __init__(self, p=0):
        torch.nn.Module.__init__(self)
        self.p = p

    def apply(self, psi: State):
        beta = torch.randn(2, dtype=tcomplex, device=psi.device)
        beta /= torch.linalg.norm(beta)
        return gate_implementation.add_qubit(self.p, beta, psi)

    def apply_reverse(self, psi: State):
        raise NotImplementedError

    def _reverse(self, **kwargs):
        return Measurement(self.p)

    def apply_to_density_matrix(self, rho: DensityMatrix):
        # rho_out = torch.zeros(
        #    (2 * len(rho), 2 * len(rho)), dtype=rho.dtype, device=rho.device
        # )
        # _0, _1 = gate_implementation.split_by_bit_p(2 * len(rho), self.p)
        # rho_out[_0][:, _0] += rho
        # rho_out[_1][:, _1] += rho
        # return rho_out / 2

        assert self.p == 0
        zero = torch.zeros_like(rho)
        return (
            torch.cat(
                [torch.cat([rho, zero], axis=1), torch.cat([zero, rho], axis=1)], axis=0
            )
            / 2
        )


class Measurement(nn.Module, Gate):
    implementation = config.get_default_gate_implementation()
    outcome_tuple = namedtuple("Measurement", ["psi", "outcome", "p_outcome"])
    p: int

    def __init__(self, p=0):
        torch.nn.Module.__init__(self)
        self.p = p

    def apply(self, psi: State, u: uniform01, normalize=True):
        return self.measure(psi, u, normalize=normalize)

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

    def apply_reverse(self, psi: State, outcome: bool):
        N = 2 * len(psi)
        _0, _1 = self.implementation.split_by_bit_p(N, self.p)
        psi_out = torch.zeros(N, dtype=psi.dtype, device=psi.device)
        psi_out[[_0, _1][outcome]] += psi
        return psi_out

    def _reverse(self, **kwargs):
        return AddAncilla(self.p)

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho: DensityMatrix):
        return self.partial_trace(rho)

    def partial_trace(self, rho: DensityMatrix):
        _0, _1 = self.implementation.split_by_bit_p(len(rho), self.p)
        out = rho[_0][:, _0] + rho[_1][:, _1]
        return out
