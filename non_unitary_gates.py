from differentiable_gate import *
import torch.nn as nn


class SingleQubitGate(Gate, nn.Module):
    k = 1

    def __init__(self, p):
        Gate.__init__(self, (p,))
        nn.Module.__init__(self)

    def p(self):
        return self.positions[0]


class AddAncilla(SingleQubitGate):
    def apply(self, psi: State):
        N = 2 * len(psi)
        psi_out = torch.zeros((N,) + psi.shape[1:], dtype=psi.dtype, device=psi.device)
        _0, _1 = gate_implementation.split_by_bits(N, [self.p()])
        psi_out[_0] += psi
        return psi_out

    def apply_reverse(self, psi: State):
        return RestrictMeasurementOutcome.apply(self, psi)

    # def _reverse(self, **kwargs):
    #     return RestrictMeasurementOutcome(self.positions)


class RestrictMeasurementOutcome(AddAncilla):
    def apply(self, psi: State):
        _0, _1 = gate_implementation.split_by_bits(len(psi), [self.p()])
        return psi[_0]

    def apply_reverse(self, psi: State):
        return AddAncilla.apply(self, psi)

    # def _reverse(self, **kwargs):
    #     return AddAncilla(self.positions)

    # def apply_to_density_matrix(self, rho: DensityMatrix):
    #    _0, _1 = gate_implementation.split_by_bit_p(len(rho), self.p)
    #    return rho[_0][:, _0]


# class AddRandomAncilla(Gate, torch.nn.Module):
#    p: int
#
#    def __init__(self, p=0):
#        torch.nn.Module.__init__(self)
#        self.p = p


class AddRandomAncilla(SingleQubitGate):
    def apply(self, psi: State):
        beta = torch.randn(2, dtype=tcomplex, device=psi.device)
        beta /= torch.linalg.norm(beta)
        return gate_implementation.add_qubits(self.positions, beta, psi)

    def apply_reverse(self, psi: State):
        raise NotImplementedError

    # def _reverse(self, **kwargs):
    #     return Measurement(self.p)

    def apply_to_density_matrix(self, rho: DensityMatrix):
        # rho_out = torch.zeros(
        #    (2 * len(rho), 2 * len(rho)), dtype=rho.dtype, device=rho.device
        # )
        # _0, _1 = gate_implementation.split_by_bit_p(2 * len(rho), self.p)
        # rho_out[_0][:, _0] += rho
        # rho_out[_1][:, _1] += rho
        # return rho_out / 2

        assert self.p() == 0
        zero = torch.zeros_like(rho)
        return (
            torch.cat(
                [torch.cat([rho, zero], axis=1), torch.cat([zero, rho], axis=1)], axis=0
            )
            / 2
        )


# class Measurement(nn.Module, Gate):
#    implementation = config.get_default_gate_implementation()
#    outcome_tuple = namedtuple("Measurement", ["psi", "outcome", "p_outcome"])
#    p: int
#
#    def __init__(self, p=0):
#        torch.nn.Module.__init__(self)
#        self.p = p


class Measurement(SingleQubitGate):
    outcome_tuple = namedtuple("Measurement", ["psi", "outcome", "p_outcome"])

    def apply(self, psi: State, u: uniform01, normalize=True):
        return self.measure(psi, u, normalize=normalize)

    def measure(self, psi: State, u: uniform01, normalize=True):
        _0, _1 = gate_implementation.split_by_bits(len(psi), self.positions)
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
        _0, _1 = gate_implementation.split_by_bits(N, self.positions)
        psi_out = torch.zeros(N, dtype=psi.dtype, device=psi.device)
        psi_out[[_0, _1][outcome]] += psi
        return psi_out

    # def _reverse(self, **kwargs):
    #     return AddAncilla(self.positions)

    """
    Test utilities.
    """

    def apply_to_density_matrix(self, rho: DensityMatrix):
        return self.partial_trace(rho)

    def partial_trace(self, rho: DensityMatrix):
        _0, _1 = gate_implementation.split_by_bits(len(rho), self.positions)
        out = rho[_0][:, _0] + rho[_1][:, _1]
        return out
