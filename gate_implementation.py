import torch
from datatypes import GateImplementation
from config import tcomplex
import config
from collections import namedtuple


def bit_p(N, p):
    blocksize = N // 2 ** (p + 1)
    arange = torch.arange(N, device=config.device)
    b = (arange // blocksize) % 2
    return b, blocksize


def split_by_bit_p(N, p):
    b, blocksize = bit_p(N, p)
    (_0,) = torch.where(b == 0)
    _1 = _0 + blocksize
    return _0, _1


def split_by_bits_pq(N, p, q):
    bp, blocksize_p = bit_p(N, p)
    bq, blocksize_q = bit_p(N, q)
    (_00,) = torch.where((bp == 0) * (bq == 0))
    _01 = _00 + blocksize_q
    _10 = _00 + blocksize_p
    _11 = _00 + (blocksize_p + blocksize_q)
    return _00, _01, _10, _11


class TorchGate(GateImplementation):
    @staticmethod
    def split_by_bit_p(N, p):
        return split_by_bit_p(N, p)

    @staticmethod
    def apply_gate(gate, gate_state, psi):
        if gate.k == 1:
            index_arrays = split_by_bit_p(len(psi), gate.p)
        elif gate.k == 2:
            index_arrays = split_by_bits_pq(len(psi), gate.p, gate.q)

        psi_out = torch.zeros_like(psi, device=config.device, dtype=tcomplex)
        if gate.diag:
            for i, I in enumerate(index_arrays):
                psi_out[I] = gate_state[i] * psi[I]
        else:
            for i, I in enumerate(index_arrays):
                for j, J in enumerate(index_arrays):
                    psi_out[I] += gate_state[i, j] * psi[J]
        del psi
        return psi_out


class EvolveDensityMatrix(TorchGate):
    """
    For testing
    """

    @staticmethod
    def apply_gate(gate, gate_state, rho):
        M_rho = TorchGate.apply_gate(gate, gate_state, rho)
        M_rho_Mt = TorchGate.apply_gate(gate, gate_state, M_rho.T.conj())
        return M_rho_Mt
