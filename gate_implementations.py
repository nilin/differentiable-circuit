import torch
from differentiable_gate import GateImplementation
from config import tcomplex
import config
from collections import namedtuple


def bit_p(N, p, device):
    blocksize = N // 2 ** (p + 1)
    arange = torch.arange(N, device=device)
    b = (arange // blocksize) % 2
    return b, blocksize


def split_by_bit_p(N, psi, p, device):
    b, blocksize = bit_p(N, p, device)
    (_0,) = torch.where(b == 0)
    _1 = _0 + blocksize
    return _0, _1


def split_by_bits_pq(N, psi, p, q, device):
    bp, blocksize_p = bit_p(N, p, device)
    bq, blocksize_q = bit_p(N, q, device)
    (_00,) = torch.where((bp == 0) * (bq == 0))
    _01 = _00 + blocksize_q
    _10 = _00 + blocksize_p
    _11 = _00 + (blocksize_p + blocksize_q)
    return _00, _01, _10, _11


class TorchGate(GateImplementation):
    device: torch.device = torch.device(config.device)

    def apply_gate(self, gate, gate_state, psi):
        if gate.k == 1:
            index_arrays = split_by_bit_p(len(psi), psi, gate.p, self.device)
        elif gate.k == 2:
            index_arrays = split_by_bits_pq(len(psi), psi, gate.p, gate.q, self.device)

        psi_out = torch.zeros_like(psi, device=self.device, dtype=tcomplex)
        if gate.diag:
            for i, I in enumerate(index_arrays):
                psi_out[I] = gate_state[i] * psi[I]
        else:
            for i, I in enumerate(index_arrays):
                for j, J in enumerate(index_arrays):
                    psi_out[I] += gate_state[i, j] * psi[J]
        del psi
        return psi_out


def torchcomplex(x):
    real = torch.Tensor(x.real)
    imag = torch.Tensor(x.imag)
    return torch.complex(real, imag)
