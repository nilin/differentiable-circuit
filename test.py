from config import device
import config
import numpy as np
import torch
import argparse
from differentiable_circuit import Params, cdot, Circuit, squared_overlap
from typing import Literal
from typing import List, Callable
import gates_and_circuits
from gates_and_circuits import Block, Lindblad, zero_state
import copy
from gate_implementation import TorchGate, EvolveDensityMatrix, GateImplementation
from torch.nn import Parameter


def re_implement(circuit: Circuit, implementation: GateImplementation):
    for gate in circuit.gates:
        gate.implementation = implementation


class TestGrad:
    def __init__(self, n=4):
        self.params = Parameter(torch.Tensor([1.0] * 4))
        a, b, c, d = self.params
        B1 = Block(n, a, b)
        B2 = Block(n, c, d)
        self.circuit = Circuit(gates=B1.gates + B2.gates)
        self.psi0 = zero_state(n)
        self.target = zero_state(n)

    def optimal_control_grad(self):
        def Obs(y):
            return self.target * cdot(self.target, y)

        self.circuit.optimal_control(self.psi0, Obs)
        return self.params.grad

    def auto_grad(self):
        loss = squared_overlap(self.target, self.circuit.apply(self.psi0))
        loss.backward()
        return self.params.grad

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()

        re_implement(self.circuit, EvolveDensityMatrix())
        rho_out = self.circuit.apply(rho)
        re_implement(self.circuit, TorchGate())

        loss = cdot(self.target, rho_out @ self.target).real
        loss.backward()
        return self.params.grad


class TestGradChannel(TestGrad):
    def __init__(self, n):
        self.params = Parameter(torch.Tensor([1.0] * 4))
        a, b, c, d = self.params
        B1 = Block(n, a, b)
        B2 = Block(n, c, d)
        self.circuit = Lindblad(B1, B2)
        self.psi0 = zero_state(n)
        self.target = zero_state(n)


def average(get_grad, n):
    from tqdm import tqdm

    randomness = np.random.uniform(0, 1, (n, 2))
    sum = 0
    # for i, rand in tqdm(enumerate(randomness)):
    for i, rand in enumerate(randomness):
        sample = get_grad(randomness=rand)
        sum += sample
        if i % 100 == 0:
            print(sum / (i + 1))

    return sum / n


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=4)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    print(TestGrad().optimal_control_grad())
    print(TestGrad().auto_grad())
    print(TestGrad().density_matrix_grad())

    # print("\nquantum control gradient  ", test_grad_unitary.get_grad())
    # print("\nsanity check (param shift)", test_grad_unitary.get_grad_paramshift())

    # test_grad_channel = TestGradChannel()
    # print("\nsampling channel for quantum control gradient")
    # print(average(test_grad_channel.get_grad, 1000))
    # print("\nsampling channel for param shift sanity check")
    # print(average(test_grad_channel.get_grad_paramshift, 1000))
