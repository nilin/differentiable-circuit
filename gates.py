import globalconfig
from differentiable_circuit import Gate, AutoGate
import jax.numpy as jnp
import numpy as np

from apply_gate_numba import (
    apply_gate_1q as apply_gate_1q_numba,
    apply_gate_2q as apply_gate_2q_numba,
    apply_diag_2q as apply_diag_2q_numba,
)
from apply_gate_torch import (
    apply_gate_1q as apply_gate_1q_torch,
    apply_gate_2q as apply_gate_2q_torch,
    apply_diag_2q as apply_diag_2q_torch,
)


class SwitchGate(Gate):
    def apply(self, gate, psi, implementation=None, **kwargs):
        if hasattr(self, "p") and "p" not in kwargs:
            kwargs["p"] = self.p

        if hasattr(self, "q") and "q" not in kwargs:
            kwargs["q"] = self.q

        if implementation == "numba":
            args = []
            if "p" in kwargs:
                args.append(kwargs["p"])
            if "q" in kwargs:
                args.append(kwargs["q"])

            flatgate = np.array(gate, dtype=np.complex64).flatten()
            psi_out = np.zeros_like(psi, dtype=np.complex64)
            self.apply_fns["numba"](psi_out, psi, flatgate, *args, len(psi))
            return psi_out

        if implementation == "torch":
            gate = np.array(gate, dtype=np.complex64)
            psi = self.apply_fns["torch"](psi, gate, **kwargs)
            return psi


def test(*a, **kw):
    print(a)
    print(kw)


class Gate_1q(SwitchGate):
    apply_fns = {
        "numba": apply_gate_1q_numba,
        "torch": apply_gate_1q_torch,
    }


class Gate_2q(SwitchGate):
    apply_fns = {
        "numba": apply_gate_2q_numba,
        "torch": apply_gate_2q_torch,
    }


class Gate_2q_diag(SwitchGate):
    apply_fns = {
        "numba": apply_diag_2q_numba,
        "torch": apply_diag_2q_torch,
    }

    def inverse(self, gate):
        return gate.conj()


class SingleParamGate(AutoGate):
    def gate(self, t):
        (t,) = t
        return self._gate(t)


class UX(SingleParamGate, Gate_1q):
    def _gate(self, t):
        return jnp.array(
            [
                [jnp.cos(t), -1j * jnp.sin(t)],  # check if minus
                [-1j * jnp.sin(t), jnp.cos(t)],  #
            ]
        )


class UZZ(SingleParamGate, Gate_2q_diag):
    def _gate(self, t):
        w = jnp.exp(1j * t)
        w_ = w.conj()
        return jnp.array((w_, w, w, w_), dtype=jnp.complex64)


class UA(SingleParamGate, Gate_2q):
    def _gate(self, t):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        XZ = np.kron(X, Z)
        eigs, U = np.linalg.eigh(XZ)

        U = jnp.array(U)
        D = jnp.exp(-1j * t * eigs)
        return U @ (D[:, None] * U.T)
