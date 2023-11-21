import torch
from typing import List, Tuple, Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcomplex = torch.complex64


"""obtain indices"""


def split_by_bits_fn(N: int, positions: List[int]):
    where_all_k_qubits_0, blocksizes = get_where_all_k_qubits_0(N, positions)

    def get_indices(i):
        shift = dot_with_binary_expansion(blocksizes, i)
        return where_all_k_qubits_0 + shift

    return get_indices


def split_by_bits(N: int, positions: List[int]):
    k = len(positions)
    where_all_k_qubits_0, blocksizes = get_where_all_k_qubits_0(N, positions)

    for i in range(2**k):
        shift = dot_with_binary_expansion(blocksizes, i)
        yield where_all_k_qubits_0 + shift


"""helper functions for obtaining indices"""


def bit_p(N: int, p: int):
    blocksize = N // 2 ** (p + 1)
    arange = torch.arange(N, device=device)
    b = (arange // blocksize) % 2
    return b, blocksize


def get_where_all_k_qubits_0(N: int, positions: List[int]):
    are_all_k_qubits_0 = torch.ones(N, dtype=torch.bool, device=device)
    blocksizes = []
    for p in positions:
        bit, blocksize = bit_p(N, p)
        are_all_k_qubits_0 = are_all_k_qubits_0 * (bit == 0)
        blocksizes.append(blocksize)

    (where_all_k_qubits_0,) = torch.where(are_all_k_qubits_0)
    return where_all_k_qubits_0, blocksizes


def dot_with_binary_expansion(array, i: int):
    overlap = 0
    for j, element in enumerate(array[::-1]):
        overlap += element * ((i >> j) & 1)
    return overlap


"""apply gates"""


def apply_gate(positions: List[int], k_qubit_matrix: torch.Tensor, psi: torch.Tensor):
    N = len(psi)
    psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)

    for i, I in enumerate(split_by_bits(N, positions)):
        for j, J in enumerate(split_by_bits(N, positions)):
            if k_qubit_matrix[i, j] != 0:
                psi_out[I] += k_qubit_matrix[i, j] * psi[J]
            del J
        del I

    del psi
    return psi_out


def apply_sparse_gate(positions: List[int], k_qubit_sparse_matrix: Tuple, psi: torch.Tensor):
    N = len(psi)
    psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)
    matrix_indices, values = k_qubit_sparse_matrix
    get_indices = split_by_bits_fn(N, positions)

    for (i, j), val in zip(matrix_indices, values):
        I = get_indices(i)
        J = get_indices(j)
        psi_out[I] += val * psi[J]
        del I
        del J

    del psi
    return psi_out


def apply_gate_diag(positions: List[int], k_qubit_diag: torch.Tensor, psi: torch.Tensor):
    """version 1"""
    N = len(psi)
    psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)

    for i, I in enumerate(split_by_bits(N, positions)):
        if k_qubit_diag[i] != 0:
            psi_out[I] += k_qubit_diag[i] * psi[I]
        del I

    del psi
    return psi_out

    """version 2"""
    # i = torch.arange(len(k_qubit_diag))
    # indices = torch.stack([i, i], axis=1)
    # sparse = (indices, k_qubit_diag)
    # return apply_sparse_gate(positions, sparse, psi)


"""changing system size"""


def apply_on_complement(exclude_positions: Tuple, gate_fn: Callable, psi: torch.Tensor):
    psi_out = torch.zeros_like(psi)
    for I in split_by_bits(len(psi), exclude_positions):
        psi_out[I] = gate_fn(psi[I])
        del I

    del psi
    return psi_out


def add_qubits(positions, beta, psi):
    k = len(positions)
    psi_out = torch.zeros(2 * len(psi), device=device, dtype=tcomplex)
    for i, I in enumerate(split_by_bits(2**k * len(psi), positions)):
        psi_out[I] = beta[i] * psi
        del I

    del psi
    return psi_out
