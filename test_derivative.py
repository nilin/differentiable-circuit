from derivative import *
import time
import torch

gate = UX()

L = 16
psi = np.arange(2**L, dtype=np.complex64)
_gate = [(0, 1), (1, 0)]

psi = gate.apply(_gate, psi, p=1, implementation="numba")

t0 = time.time()

for _ in range(10):
    for p in range(L):
        y, dy, dt = gate.forward_parameterized(
            0.1, psi, psi, p=p, implementation="numba"
        )
        print("numba", p)

t1 = time.time()
print("numba time", t1 - t0)
print()

psi = torch.Tensor(psi)

for _ in range(10):
    for p in range(L):
        y, dy, dt = gate.forward_parameterized(
            0.1, psi, psi, p=p, implementation="torch"
        )
        print("torch", p)

t2 = time.time()
print("torch time", t2 - t1)

print(psi)
print(y)
print(dy)
print(dt)
