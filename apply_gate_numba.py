from globalconfig import *
import numba
from numba import jit, njit, prange
from numba import cuda
import numba as nb
import numpy as np
from numba import boolean as bool, uint8 as u, uint64 as U, void, complex64 as C, types
from numba.types import Tuple
from cudaswitch import new_value_at_i, cudaswitch_device


@cudaswitch_device(
    numba.types.Tuple((numba.boolean, numba.uint64))(
        numba.uint64,
        numba.uint8,
        numba.uint64,
    )
)
def bitvalue(N, p, i):
    blocksize = N // 2 ** (p + 1)
    bit = (i // blocksize) % 2
    return int(bit), int(blocksize)


@new_value_at_i(
    numba.void(
        numba.complex64[:],
        numba.complex64[:],
        numba.complex64[:],
        numba.uint8,
        numba.uint64,
    )
)
def apply_gate_1q(psi_out, psi_in, flatgate, p, i):
    b_out, blocksize = bitvalue(len(psi_in), p, i)

    r0, r1 = flatgate[2 * b_out], flatgate[2 * b_out + 1]
    i0 = i - blocksize * b_out

    psi_out[i] = r0 * psi_in[i0] + r1 * psi_in[i0 + blocksize]


@new_value_at_i(
    numba.void(
        numba.complex64[:],
        numba.complex64[:],
        numba.complex64[:],
        numba.uint8,
        numba.uint8,
        numba.uint64,
    )
)
def apply_gate_2q(psi_out, psi_in, flatgate, p, q, i):
    bp, blocksize_p = bitvalue(len(psi_in), p, i)
    bq, blocksize_q = bitvalue(len(psi_in), q, i)

    i0 = i - blocksize_p * bp - blocksize_q * bq
    rownumber = 2 * bp + bq

    psi_out[i] = 0
    for jp in range(2):
        for jq in range(2):
            j = 2 * jp + jq
            i1 = i0 + blocksize_p * jp + blocksize_q * jq
            psi_out[i] += flatgate[4 * rownumber + j] * psi_in[i1]


@new_value_at_i(
    numba.void(
        numba.complex64[:],
        numba.complex64[:],
        numba.complex64[:],
        numba.uint8,
        numba.uint8,
        numba.uint64,
    )
)
def apply_diag_2q(psi_out, psi, gate, p, q, i):
    bp, _ = bitvalue(len(psi), p, i)
    bq, _ = bitvalue(len(psi), q, i)

    entry = 2 * bp + bq
    psi_out[i] = gate[entry] * psi[i]


@new_value_at_i(
    numba.void(
        numba.complex64[:],
        numba.uint8,
        numba.uint64,
    )
)
def Sum_ZZ(reg, L, i):
    N = 2**L

    reg[i] = 0
    for p in range(L - 1):
        b1, _ = bitvalue(N, p, i)
        b2, _ = bitvalue(N, p + 1, i)
        if b1 == b2:
            reg[i] += 1
        else:
            reg[i] -= 1
