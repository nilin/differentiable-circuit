from typing import Any
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Tuple, Union


# def apply_Z_basis_gate(full_gate, psi):


class Gate:
    def __init__(self, param_names, param_transform=None, p=None, q=None):
        self.param_names = param_names
        self.param_transform = param_transform
        if self.param_transform is None:
            self.param_transform = ", ".join(self.param_names)
        if p is not None:
            self.p = p
        if q is not None:
            self.q = q

    def _tangent_(self, gate, Dgate, x, dx, **kwargs):
        apply = partial(self.apply, **kwargs)

        y = apply(gate, x)
        dy = apply(gate, dx)
        dthetas = []
        for dgate in Dgate:
            dy_gate = apply(dgate, x)
            dthetas.append(np.dot(dy, dy_gate))
        return y, dy, dthetas

    def tangent(self, thetas, x, dx, reverse=False, **kwargs):
        gate, Dgate = self.val_grad_gate(thetas, reverse=reverse)
        return self._tangent_(gate, Dgate, x, dx, **kwargs)

    def apply_parameterized(self, thetas, x, reverse=False, **kwargs):
        gate, Dgate = self.val_grad_gate(thetas, reverse=reverse)
        return self.apply(gate, x, **kwargs)

    def val_grad_gate(self, thetas, reverse=False):
        raise NotImplementedError

    def apply(self, gate, psi, **kwargs):
        raise NotImplementedError

    def get_array(self, t, reverse=False):
        return self.val_grad_gate(t, reverse=reverse)[0]

    @staticmethod
    def inverse(gate):
        inv = gate.conj().T
        return inv


class AutoGate(Gate):
    def apply_transform(self, t):
        param_names = ", ".join(self.param_names)
        lambdadef = f"lambda {param_names}: " + self.param_transform
        f = eval(lambdadef)
        return f(t)

    def val_grad_gate(self, t, reverse=False):
        if reverse:
            f = self.inverse
        else:
            f = lambda x: x

        _gate = lambda t: f(self.gate(self.apply_transform(t)))
        _grad_gate = jax.jacfwd(_gate)
        return (
            np.array(_gate(t), dtype=np.complex64),
            np.array(_grad_gate(t), dtype=np.complex64),
        )


class Measure01(Gate):
    param_names = []

    def __init__(self, name, p):
        self.name = name
        self.p = p

    def val_grad_gate(self, thetas, reverse=False):
        return np.zeros(0), []

    def apply(self, gate, psi, randomness, **kwargs):
        p = self.p
        u = randomness[self.name]
        a, b = self.get_01_weights(psi, p)
        separator = a / (a + b)
        outcome = u > separator
        return self.setbit(psi, outcome, p)

    @staticmethod
    def get_01_weights(psi, p):
        raise NotImplementedError

    @staticmethod
    def setbit(psi, value, p):
        raise NotImplementedError


class UnitaryCircuit:
    def __init__(self, gates):
        self.gates: List[Gate] = gates

    def loss_and_grad(self, lossfn_and_grad, theta_dict, x, **kwargs):
        y = self.run(theta_dict, x, **kwargs)
        loss, dy = lossfn_and_grad(y)
        x, dx, dthetas = self.tangent(theta_dict, y, dy, reverse=True, **kwargs)
        return loss, dict(dthetas=dthetas, dx=dx)

    def tangent(self, theta_dict, x, dx, reverse=False, **kwargs):
        gates = self.gates[::-1] if reverse else self.gates

        for gate in gates:
            thetas = [theta_dict[name] for name in gate.param_names]
            x, dx, dthetas = gate.tangent(thetas, x, dx, reverse=reverse, **kwargs)
            dtheta_dict = {
                name: np.zeros_like(theta, dtype=np.complex64)
                for name, theta in theta_dict.items()
            }

            for name, dtheta in zip(gate.param_names, dthetas):
                dtheta_dict[name] += dtheta

        return x, dx, dtheta_dict

    def run(self, theta_dict, x, reverse=False, **kwargs):
        gates = self.gates[::-1] if reverse else self.gates

        for gate in gates:
            thetas = [theta_dict[name] for name in gate.param_names]
            x = gate.apply_parameterized(thetas, x, reverse, **kwargs)
        return x
