"""
https://qiskit.org/documentation/tutorials/operators/02_gradients_framework.html
"""
# General imports
from typing import Sequence
import numpy as np
from numpy import pi
from functools import partial
from qiskit.circuit.bit import Bit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.register import Register

from _test import *

# Operator Imports
from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian

# Circuit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    Parameter,
    ParameterVector,
    ParameterExpression,
)
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import EfficientSU2
from scipy.linalg import expm
