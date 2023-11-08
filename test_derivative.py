import globalconfig
from globalconfig import implementation as impl
from differentiable_circuit import *
from gates import *
import time
import torch
import numpy as np
from numpy.testing import assert_allclose
import cudaswitch
from jax.tree_util import tree_map
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--L", type=int, default=20)
args, _ = argparser.parse_known_args()


def test():
    L = args.L

    print(f"{L} qubits, vector size {2**L:,}")

    psi = np.arange(2**L, dtype=np.complex64)
    psi /= np.linalg.norm(psi)

    if impl == "numba":
        x = cudaswitch.try_to_device(psi)

    if impl == "torch":
        x = utils.torchcomplex(psi)
        x = x.to(globalconfig.torchdevice)

    nlayers = 1

    Xs = [UX(["theta"], p=i) for i in range(L)]
    ZZs = [UZZ(["phi"], p=i, q=i + 1) for i in range(L - 1)]
    layer = Xs + ZZs
    C = [op for k in range(nlayers) for op in layer]

    t0 = time.time()
    C = UnitaryCircuit(C)

    thetas = {"theta": 0.1, "phi": 0.2}

    y = C.run(thetas, x)
    z = C.run(thetas, y, reverse=True)

    if impl == "numba":
        x = torch.as_tensor(x, device=torch.device("cuda"))
        y = torch.as_tensor(y, device=torch.device("cuda"))
        z = torch.as_tensor(z, device=torch.device("cuda"))

    print("norm of output", torch.norm(y).cpu().numpy())
    print("error", torch.norm(x - z).cpu().numpy())

    t1 = time.time()
    print(f"time for {len(C.gates)} gates: {t1-t0:.3f}s")

    getupdates = lambda lr, dT: tree_map(lambda dt: -lr * dt, dthetas)
    apply = lambda T, dT: tree_map(lambda t, dt: (t + dt).real, thetas, dthetas)
    target = x
    lossfn = lambda y: torch.norm(y - target)

    for i in range(10000):
        loss, dthetas, dx = C.loss_and_grad(thetas, x, lossfn=lossfn)
        updates = getupdates(0.1, dthetas)
        thetas = apply(thetas, updates)
        print(loss.detach().cpu().numpy())


def testgrad():
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


# testcorrect()
test()
