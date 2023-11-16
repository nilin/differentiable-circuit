import torch
from datatypes import GateImplementation
from datatypes import tcomplex
import config

device = config.device


def bit_p(N, p):
    blocksize = N // 2 ** (p + 1)
    arange = torch.arange(N, device=device)
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

        psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)
        if gate.diag:
            for i, I in enumerate(index_arrays):
                psi_out[I] += gate_state[i] * psi[I]
        else:
            for i, I in enumerate(index_arrays):
                for j, J in enumerate(index_arrays):
                    psi_out[I] += gate_state[i, j] * psi[J]
        del psi
        return psi_out


class AnySizeGate(GateImplementation):
    @staticmethod
    def split_by_bits(N, positions):
        k = len(positions)

        are_all_k_qubits_0 = torch.ones(N, dtype=torch.bool, device=device)
        blocksizes = []
        for p in positions:
            bit, blocksize = bit_p(N, p)
            are_all_k_qubits_0 = are_all_k_qubits_0 * (bit == 0)
            blocksizes.append(blocksize)

        (where_all_k_qubits_0,) = torch.where(are_all_k_qubits_0)

        for i in range(2**k):
            shift = AnySizeGate.dot_with_binary_expansion(blocksizes, i)
            indices = where_all_k_qubits_0 + shift
            yield indices

    @staticmethod
    def dot_with_binary_expansion(array, i):
        overlap = 0
        for j, element in enumerate(array[::-1]):
            overlap += element * ((i >> j) & 1)
        return overlap

    @staticmethod
    def apply_gate_to_qubits(positions, k_qubit_matrix, psi, diag=False):
        N = len(psi)

        psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)

        if diag:
            for i, I in enumerate(AnySizeGate.split_by_bits(N, positions)):
                psi_out[I] += k_qubit_matrix[i] * psi[I]

        else:
            for i, I in enumerate(AnySizeGate.split_by_bits(N, positions)):
                for j, J in enumerate(AnySizeGate.split_by_bits(N, positions)):
                    psi_out[I] += k_qubit_matrix[i, j] * psi[J]

        del psi
        return psi_out

    @staticmethod
    def apply_gate(gate, gate_state, psi):
        return AnySizeGate.apply_gate_to_qubits(
            gate.positions, gate_state, psi, gate.diag
        )
