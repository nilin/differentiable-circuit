import numba
from numba import jit, njit, prange
from numba import cuda
from cudaswitch import new_value_at_i


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
    N = len(psi_out)
    blocksize = N // 2 ** (p + 1)
    b_out = (i // blocksize) % 2

    if b_out:
        r0, r1 = flatgate[2], flatgate[3]
        i0 = i - blocksize
    else:
        r0, r1 = flatgate[0], flatgate[1]
        i0 = i

    i0 = int(i0)
    blocksize = int(blocksize)
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
    N = len(psi_out)
    blocksize_p = N // 2 ** (p + 1)
    blocksize_q = N // 2 ** (q + 1)
    bp = (i // blocksize_p) % 2
    bq = (i // blocksize_q) % 2

    i0 = i - blocksize_p * bp - blocksize_q * bq
    rownumber = 2 * bp + bq

    i0 = int(i0)
    blocksize = int(blocksize)
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
        numba.uint8,
        numba.uint8,
        numba.uint64,
    )
)
def apply_diag_2q(psi, gate, p, q, i):
    N = len(psi)
    blocksize_p = N // 2 ** (p + 1)
    blocksize_q = N // 2 ** (q + 1)
    bp = (i // blocksize_p) % 2
    bq = (i // blocksize_q) % 2

    entry = 2 * bp + bq
    psi[i] = gate[entry] * psi[i]
