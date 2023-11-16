import torch
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcomplex = torch.complex64


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


def split_by_bits(N: int, positions: List[int]):
    k = len(positions)
    where_all_k_qubits_0, blocksizes = get_where_all_k_qubits_0(N, positions)
    for i in range(2**k):
        shift = dot_with_binary_expansion(blocksizes, i)
        yield where_all_k_qubits_0 + shift


def dot_with_binary_expansion(array, i: int):
    overlap = 0
    for j, element in enumerate(array[::-1]):
        overlap += element * ((i >> j) & 1)
    return overlap


def apply_gate(positions: List[int], k_qubit_matrix: torch.Tensor, psi: torch.Tensor):
    N = len(psi)
    psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)

    for i, I in enumerate(split_by_bits(N, positions)):
        for j, J in enumerate(split_by_bits(N, positions)):
            if k_qubit_matrix[i, j] != 0:
                psi_out[I] += k_qubit_matrix[i, j] * psi[J]

    del psi
    return psi_out


def apply_sparse_gate(positions: List[int], k_qubit_sparse_matrix: Tuple, psi: torch.Tensor):
    N = len(psi)
    psi_out = torch.zeros_like(psi, device=device, dtype=tcomplex)
    matrix_indices, values = k_qubit_sparse_matrix
    where_all_k_qubits_0, blocksizes = get_where_all_k_qubits_0(N, positions)
    breakpoint()

    for (i, j), val in zip(matrix_indices, values):
        I = where_all_k_qubits_0 + dot_with_binary_expansion(blocksizes, i)
        J = where_all_k_qubits_0 + dot_with_binary_expansion(blocksizes, j)
        psi_out[I] += val * psi[J]

    del psi
    return psi_out
