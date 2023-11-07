from typing import Any
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp


# def apply_Z_basis_gate(full_gate, psi):


class Gate:
    def forward(self, t, x, dx, **kwargs):
        gate, dgate = self.val_grad_gate(t)
        return self._forward_(gate, dgate, x, dx, **kwargs)

    def _forward_(self, gate, dgate, x, dx, **kwargs):
        apply = partial(self.apply, **kwargs)

        y = apply(gate, x)
        dy = apply(gate, dx)
        dy_gate = apply(dgate, x)
        dtheta = np.dot(dy, dy_gate)
        return y, dy, dtheta


class AutoGate(Gate):
    def __init__(self, param_names, param_transform=None):
        self.param_names = param_names
        self.param_transform = param_transform
        if self.param_transform is None:
            self.param_transform = ", ".join(self.param_names)

    def apply_transform(self, t):
        param_names = ", ".join(self.param_names)
        lambdadef = f"lambda {param_names}: " + self.param_transform
        f = eval(lambdadef)
        return f(t)

    def val_grad_gate(self, t):
        _gate = lambda t: self.gate(self.apply_transform(t))
        _grad_gate = jax.jacfwd(_gate)
        return _gate(t), _grad_gate(t)


class UnitaryCircuit:
    def __init__(self, gates):
        self.gates = gates

    def forward(self, thetas, x, dx, **kwargs):
        dthetas = []
        for gate, theta in zip(self.gates, thetas):
            x, dx, dtheta = gate.forward(theta, x, dx, **kwargs)
            dthetas.append(dtheta)
        return x, dx, dthetas
