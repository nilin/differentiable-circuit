"""
https://qiskit.org/documentation/tutorials/operators/02_gradients_framework.html
"""
# General imports
import numpy as np

from _test import *

# Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, Parameter
from qiskit.quantum_info.operators import Operator


class Block(QuantumCircuit):
    L: int
    coupling_strength: float
    default_trottersteps: int
    taus: ParameterVector
    zetas: ParameterVector

    def __init__(self, L, l, coupling_strength=1.0, default_trottersteps=2):
        q = QuantumRegister(L + 1)
        super().__init__(q)
        self.L = L
        self.coupling_strength = coupling_strength
        self.default_trottersteps = default_trottersteps

        self.taus = ParameterVector("tau", l)
        self.zetas = ParameterVector("zetas", l)

        self.makeblock()
        # grad = Gradient().convert(operator=op, params=[self.taus, self.zetas])

    # gates

    def UZZ(self, i: int, j: int, t: float):
        """
        applies exp(-it Zi Zj)
        =
        diag( exp(it), exp(-it), exp(-it), exp(it) )
        """
        self.cx(i, j)
        self.rz(t, j)
        self.cx(i, j)

    def U_A(self, zeta: float, i: int = 0, j: int = 1):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        XZ = np.kron(X, Z)
        eigs, U = np.linalg.eigh(XZ)
        from_basis = Operator(U)
        to_basis = Operator(U.T)

        np.testing.assert_allclose(eigs, [-1, -1, 1, 1])

        self.unitary(to_basis, [i, j], label="U_t")
        self.rz(-zeta, i)
        self.unitary(from_basis, [i, j], label="U")

    # layers

    def U_sum_ZZ(self, t: float, start: int, end: int):
        for i in range(start, end - 1):
            self.UZZ(i, i + 1, t)

    def U_sum_X(self, t: float, start: int, end: int):
        for i in range(start, end):
            self.rx(t, i)

    def U_TFIM(self, t, trottersteps=None):
        if trottersteps is None:
            trottersteps = self.default_trottersteps
        SuzukiTrotter(
            [self.U_sum_ZZ, self.U_sum_X],
            [-t, -t * self.coupling_strength],
            trottersteps,
            start=1,
            end=self.L + 1,
        )

    def makeblock(self, trottersteps=None):
        for tau, zeta in zip(self.taus, self.zetas):
            self.U_TFIM(tau, trottersteps)
            self.U_A(zeta)


def Trotter(U_Hi_list, Ts, steps=1, **kwargs):
    for i in range(steps):
        for U_Hi, T in zip(U_Hi_list, Ts):
            U_Hi(T / steps, **kwargs)


def SuzukiTrotter(U_Hi_list, Ts, steps=2, **kwargs):
    t1, t2 = [T / steps for T in Ts]
    U1, U2 = U_Hi_list

    U1(t1 / 2, **kwargs)
    U2(t2, **kwargs)
    Trotter(U_Hi_list, Ts, steps=steps - 1, **kwargs)
    U1(t1 / 2, **kwargs)
