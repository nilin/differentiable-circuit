from derivative import *


gate = UX()

psi = np.arange(16, dtype=np.complex64)
slate = np.zeros_like(psi)
_gate = [(0, 1), (1, 0)]

psi = gate.apply(_gate, psi, slate=slate, p=1)

# dpsi = gate.forward(_gate, _gate, psi, psi, slate=slate, p=1)
y, dy, dt = gate.forward_parameterized(0.1, psi, psi, slate=slate, p=1)


print(psi)
print(y)
print(dy)
print(dt)
