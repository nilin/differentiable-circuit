from block_as_qiskit import Block
from packages import *

from qiskit.algorithms.gradients import (
    ParamShiftEstimatorGradient,
    BaseQGT,
    BaseEstimatorGradient,
)
from qiskit.primitives import Estimator


n = 4

block = Block(n + 1, 2)
print(block.draw())

# op = ~StateFn(H) @ CircuitStateFn(primitive=block, coeff=1.0)


# op = CircuitStateFn(primitive=block, coeff=1.0)
grad = ParamShiftEstimatorGradient(Estimator())

# grad = Gradient().convert(operator=op, params=[block.taus, block.zetas])

params = np.array([[1, 2], [3, 4]])
value_dict = {block.taus: params[0, :], block.zetas: params[1, :]}

grad_result = grad.assign_parameters(value_dict).eval()

breakpoint()
quit()

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

backend = BasicAer.get_backend("unitary_simulator")
job = backend.run(transpile(qc, backend))

print(job.result())
breakpoint()

# ZZ = job.result().get_unitary(qc, decimals=3)
# print(np.angle(ZZ))
# breakpoint()
##################################################

quit()

# Instantiate the quantum state
a = Parameter("a")
b = Parameter("b")
q = QuantumRegister(1)
qc = QuantumCircuit(q)
qc.h(q)
qc.rz(a, q[0])
qc.rx(b, q[0])

# Instantiate the Hamiltonian observable
H = (2 * X) + Z

# Combine the Hamiltonian observable and the state
op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.0)

# Print the operator corresponding to the expectation value
print(op)

params = [a, b]

# Define the values to be assigned to the parameters
value_dict = {a: np.pi / 4, b: np.pi}

# Convert the operator and the gradient target params into the respective operator
grad = Gradient().convert(operator=op, params=params)

# Print the operator corresponding to the Gradient
print(grad)

# Assign the parameters and evaluate the gradient
grad_result = grad.assign_parameters(value_dict).eval()
print("Gradient", grad_result)
