import numpy as np
from functools import partial
import jax
import jax.numpy as jnp


# def apply_Z_basis_gate(full_gate, psi):


class Gate:
    def forward_parameterized(self, t, x, dx, **kwargs):
        gate, dgate = self.val_grad_gate(t)
        return self.forward(gate, dgate, x, dx, **kwargs)

    def forward(self, gate, dgate, x, dx, **kwargs):
        apply = partial(self.apply, **kwargs)

        y = apply(gate, x)
        dy = apply(gate, dx)
        dy_gate = apply(dgate, x)
        dtheta = np.dot(dy, dy_gate)
        return y, dy, dtheta


class AutoGate(Gate):
    def __init__(self):
        self.gate = jax.jit(self.gate)
        self.grad_gate = jax.jit(jax.jacfwd(self.gate))

    def val_grad_gate(self, t):
        return self.gate(t), self.grad_gate(t)
