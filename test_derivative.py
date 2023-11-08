import globalconfig
from differentiable_circuit import *
from gates import *
import time
import torch
import numpy as np
from numpy.testing import assert_allclose


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

    # x = torch.Tensor(x)
    # x = torch.complex(x, x)
    # y = C.run(thetas, x, implementation="torch")
    # z = C.run(thetas, y, implementation="torch", reverse=True)

    # print(x)
    # print(np.linalg.norm(x - y))
    # print(np.linalg.norm(x - z))

    lossfn = lambda y: jnp.linalg.norm(y - target)

    def lossfn_and_grad(x):
        # x = jnp.array(x.numpy().astype(np.complex64))
        val, grad = jax.value_and_grad(lossfn)(x)
        # grad_r = torch.Tensor(np.array(grad).real)
        # grad_i = torch.Tensor(np.array(grad).imag)
        # grad = torch.complex(grad_r, grad_i)
        return val, np.array(grad, dtype=np.complex64)

    loss, grad = C.loss_and_grad(lossfn_and_grad, thetas, x, implementation="numba")
    print(loss)


testcorrect()


def testperf():
    L = 16
    psi = np.arange(2**L, dtype=np.complex64)

    gate0 = UX(["theta"], p=0)
    gate1 = UX(["theta"], p=1)
    gate2 = UX(["phi"], p=2)

    C = UnitaryCircuit([gate0, gate1, gate2])
    x = psi
    z = np.zeros_like(psi)
    y, dy, dthetas = C.perturb_forward(
        {"theta": 0.1, "phi": 0.2}, x, z, implementation="numba"
    )
    w, dw, dthetas = C.perturb_forward(
        {"theta": 0.1, "phi": 0.2}, y, z, implementation="numba"
    )

    breakpoint()


# _gate = [(0, 1), (1, 0)]
# psi = gate.apply(_gate, psi, p=1, implementation="numba")
quit()

t0 = time.time()

for _ in range(10):
    for p in range(L):
        y, dy, dt = gate.forward(0.1, psi, psi, p=p, implementation="numba")
        print("numba", p)

t1 = time.time()
print("numba time", t1 - t0)
print()

psi = torch.Tensor(psi)

for _ in range(10):
    for p in range(L):
        y, dy, dt = gate.forward(0.1, psi, psi, p=p, implementation="torch")
        print("torch", p)

t2 = time.time()
print("torch time", t2 - t1)

print(psi)
print(y)
print(dy)
print(dt)
