import globalconfig
from differentiable_circuit import Gate, AutoGate
import differentiable_circuit
import jax.numpy as jnp
import numpy as np
import torch
import apply_gate_numba
import apply_gate_torch
import cudaswitch, numba

from apply_gate_numba import (
    apply_gate_1q as apply_gate_1q_numba,
    apply_gate_2q as apply_gate_2q_numba,
    apply_diag_2q as apply_diag_2q_numba,
    Sum_ZZ as Sum_ZZ_numba,
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


class SingleParamGate(Gate):
    def gate(self, t):
        (t,) = t
        return self._gate(t)


class UX(SingleParamGate, Gate_1q, AutoGate):
    def _gate(self, t):
        return jnp.array(
            [
                [jnp.cos(t), -1j * jnp.sin(t)],  # check if minus
                [-1j * jnp.sin(t), jnp.cos(t)],  #
            ]
        )


class UZZ(SingleParamGate, Gate_2q_diag, AutoGate):
    def _gate(self, t):
        w = jnp.exp(1j * t)
        w_ = w.conj()
        return jnp.array((w_, w, w, w_), dtype=jnp.complex64)


class UA(SingleParamGate, Gate_2q, AutoGate):
    def _gate(self, t):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        XZ = np.kron(X, Z)
        eigs, U = np.linalg.eigh(XZ)

        U = jnp.array(U)
        D = jnp.exp(-1j * t * eigs)
        return U @ (D[:, None] * U.T)


class U_global_diag_H(SingleParamGate):
    def apply(self, gate, psi, **kwargs):
        pass

    def val_grad_gate(self, gate, psi, **kwargs):
        pass


class U_Sum_ZZ(SingleParamGate):
    def __init__(self, L, param_names, param_transform=None):
        super().__init__(param_names, param_transform)
        self.diag_H = np.zeros(L**2, dtype=np.complex64)
        Sum_ZZ_numba(self.diag_H, L, L**2)


class Measure01(differentiable_circuit.Measure01):
    @staticmethod
    def get_01_weights(psi, p):
        psi = torch.utils.dlpack.from_dlpack(psi)
        prob = psi * psi.conj()
        I0, I1 = apply_gate_torch.indices(psi, p)
        return prob[I0].sum(), prob[I1].sum()

    @staticmethod
    def setbit(psi, value, p):
        psi = torch.utils.dlpack.from_dlpack(psi)
        psi_out = torch.zeros_like(psi)
        I0, I1 = apply_gate_torch.indices(psi, p)
        if value == 0:
            psi_out[I0] = psi[I0]
        if value == 1:
            psi_out[I1] = psi[I1]
        psi_out = torch.utils.dlpack.to_dlpack(psi_out)
        return psi_out
