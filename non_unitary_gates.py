from differentiable_gate import *
import torch.nn as nn


class Single_qubit_gate(Gate, nn.Module):
    k = 1

    def __init__(self, p):
        Gate.__init__(self, (p,))
        nn.Module.__init__(self)

    def p(self):
        return self.positions[0]


class Custom_directed_gate(Gate):
    forward: bool = True

    def apply(self, psi: State):
        if self.forward:
            return self.apply_forward(psi)
        else:
            return self.apply_backward(psi)

    def set_direction_forward(self):
        self.forward = True
        return self

    def set_direction_backward(self):
        self.forward = False
        return self


class Add_0_ancilla(Single_qubit_gate, Custom_directed_gate):
    def apply_forward(self, psi: State):
        N = 2 * len(psi)
        psi_out = torch.zeros((N,) + psi.shape[1:], dtype=psi.dtype, device=psi.device)
        _0 = next(gate_implementation.split_by_bits(N, self.positions))
        psi_out[_0] += psi
        return psi_out

    def apply_backward(self, psi: State):
        _0 = next(gate_implementation.split_by_bits(len(psi), self.positions))
        return psi[_0]


class Measurement(Single_qubit_gate, Custom_directed_gate):
    outcome_tuple = namedtuple("Measurement", ["psi", "outcome", "p_outcome"])

    def probability(self, psi: State, outcome):
        get_indices = gate_implementation.split_by_bits_fn(len(psi), self.positions)
        I = get_indices(outcome)
        return probabilitymass(psi[I]) / probabilitymass(psi)

    def measure(self, psi: State, u: uniform01):
        _0, _1 = gate_implementation.split_by_bits(len(psi), self.positions)
        p0 = probabilitymass(psi[_0]) / probabilitymass(psi)
        outcome = u > p0
        p_outcome = p0 if outcome == 0 else 1 - p0
        indices = [_0, _1][outcome]

        psi_post = psi[indices] / torch.sqrt(p_outcome)
        return self.outcome_tuple(psi_post, outcome, p_outcome)

    def unmeasure(self, psi: State, outcome: bool):
        N = 2 * len(psi)
        _0, _1 = gate_implementation.split_by_bits(N, self.positions)
        psi_out = torch.zeros(N, dtype=psi.dtype, device=psi.device)
        psi_out[[_0, _1][outcome]] += psi
        return psi_out

    def apply_forward(self, psi: State, u: uniform01 = None):
        if u is None:
            u = torch.rand(1, dtype=torch.float, device=psi.device)
        return self.measure(psi, u)[0]

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho: DensityMatrix):
        return self.partial_trace(rho)

    def partial_trace(self, rho: DensityMatrix):
        _0, _1 = gate_implementation.split_by_bits(len(rho), self.positions)
        out = rho[_0][:, _0] + rho[_1][:, _1]
        return out


class Random_out_ancilla(Single_qubit_gate, Custom_directed_gate):
    def apply_backward(self, psi: State):
        beta = torch.randn(2, dtype=tcomplex, device=psi.device)
        beta /= torch.linalg.norm(beta)
        return gate_implementation.add_qubits(self.positions, beta, psi)
