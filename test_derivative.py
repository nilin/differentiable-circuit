import globalconfig
from differentiable_circuit import *
from gates import *
import time
import torch
import numpy as np
from numpy.testing import assert_allclose
import cudaswitch


def testcorrect():
    L = 2
    x = np.arange(2**L, dtype=np.complex64)
    target = x[::-1]

    gate0 = UX(["theta"], p=0)
    gate1 = UZZ(["theta"], p=0, q=1)

    C = UnitaryCircuit([gate0, gate1])
    thetas = {"theta": 0.1, "phi": 0.2}

    y = C.run(thetas, x, implementation="numba")
    z = C.run(thetas, y, implementation="numba", reverse=True)
    print(np.linalg.norm(x - y))
    print(np.linalg.norm(x - z))
    print(x)

    lossfn = lambda y: jnp.linalg.norm(y - target)

    def lossfn_and_grad(x):
        val, grad = jax.value_and_grad(lossfn)(x)
        return val, np.array(grad, dtype=np.complex64)

    loss, grad = C.loss_and_grad(lossfn_and_grad, thetas, x, implementation="numba")
    print(loss)


# def testcorrect_torch():
#
#    x = torch.Tensor(x)
#    x = torch.complex(x, x)
#    y = C.run(thetas, x, implementation="torch")
#    z = C.run(thetas, y, implementation="torch", reverse=True)
#
#    print(x)
#    print(np.linalg.norm(x - y))
#    print(np.linalg.norm(x - z))
#
#    def lossfn_and_grad(x):
#        # x = jnp.array(x.numpy().astype(np.complex64))
#        val, grad = jax.value_and_grad(lossfn)(x)
#        # grad_r = torch.Tensor(np.array(grad).real)
#        # grad_i = torch.Tensor(np.array(grad).imag)
#        # grad = torch.complex(grad_r, grad_i)
#        return val, np.array(grad, dtype=np.complex64)


def testperf():
    L = 16
    psi = np.arange(2**L, dtype=np.complex64)
    psi /= np.linalg.norm(psi)

    gate0 = UX(["theta"], p=0)
    gate1 = UX(["theta"], p=1)
    gate2 = UX(["phi"], p=2)

    C = UnitaryCircuit([gate0, gate1, gate2])
    x = psi
    z = np.zeros_like(psi)

    thetas = {"theta": 0.1, "phi": 0.2}
    y = C.run(thetas, x, implementation="numba")
    z = C.run(thetas, y, implementation="numba", reverse=True)
    print(np.linalg.norm(x - y))
    print(np.linalg.norm(x - z))


testcorrect()
testperf()
