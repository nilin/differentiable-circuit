import globalconfig
from globalconfig import torchdevice as device
import jax
import jax.numpy as jnp
import utils


def indices(psi, p, onehot=False):
    N = len(psi)
    blocksize = N // 2 ** (p + 1)
    I = jnp.arange(N)
    b = (I // blocksize) % 2

    if onehot:
        return b
    else:
        (I0,) = jnp.where(b == 0)
        I1 = I0 + blocksize
        return I0, I1


def apply_gate_1q(psi, gate, p):
    I0, I1 = indices(psi, p)

    psi_out = jnp.zeros_like(psi, dtype=jnp.complex64)
    psi_out[I0] = gate[0, 0] * psi[I0] + gate[0, 1] * psi[I1]
    psi_out[I1] = gate[1, 0] * psi[I0] + gate[1, 1] * psi[I1]

    del psi
    return psi_out


def apply_gate_2q(psi, gate, p, q):
    bp = indices(psi, p, onehot=True)
    bq = indices(psi, q, onehot=True)

    I0 = torch.where((bp == 0) * (bq == 0))
    I1 = torch.where((bp == 0) * (bq == 1))
    I2 = torch.where((bp == 1) * (bq == 0))
    I3 = torch.where((bp == 1) * (bq == 1))

    psi_out = torch.zeros_like(psi, device=device, dtype=torch.complex64)
    for i, I in enumerate([I0, I1, I2, I3]):
        for j, J in enumerate([I0, I1, I2, I3]):
            if gate[i, j] != 0:
                psi_out[I] += gate[i, j] * psi[J]

    del psi
    return psi_out


def apply_diag_2q(psi, gate, p, q):
    bp = indices(psi, p, onehot=True)
    bq = indices(psi, q, onehot=True)

    I0 = torch.where((bp == 0) * (bq == 0))
    I1 = torch.where((bp == 0) * (bq == 1))
    I2 = torch.where((bp == 1) * (bq == 0))
    I3 = torch.where((bp == 1) * (bq == 1))

    psi_out = torch.zeros_like(psi, device=device, dtype=torch.complex64)
    psi_out[I0] = gate[0] * psi[I0]
    psi_out[I1] = gate[1] * psi[I1]
    psi_out[I2] = gate[2] * psi[I2]
    psi_out[I3] = gate[3] * psi[I3]

    del psi
    return psi_out