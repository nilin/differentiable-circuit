from config import device
import config
import numpy as np
import torch
import argparse
from differentiable_circuit import cdot, Circuit, squared_overlap
from typing import Literal
from typing import List, Callable
import gates_and_circuits
from functools import partial
from gates_and_circuits import Block, Lindblad, zero_state
import copy
from gate_implementation import TorchGate, EvolveDensityMatrix, GateImplementation
from torch.nn import Parameter


class TestGrad:
    def __init__(self, n=4):
        self.params = Parameter(torch.Tensor([1.0] * 4))
        a, b, c, d = self.params
        B1 = Block(n, a, b)
        B2 = Block(n, c, d)
        self.circuit = Circuit(gates=B1.gates + B2.gates)
        self.psi0 = zero_state(n)
        self.target = zero_state(n)

    def optimal_control_grad(self, **kwargs):
        def Obs(y):
            return self.target * cdot(self.target, y)

        self.circuit.optimal_control(self.psi0, Obs, **kwargs)
        return self.params.grad

    def auto_grad(self):
        loss = squared_overlap(self.target, self.circuit.apply(self.psi0))
        loss.backward()
        return self.params.grad

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()

        rho_out = self.circuit.apply_to_density_matrix(rho)

        loss = cdot(self.target, rho_out @ self.target).real
        loss.backward()
        return self.params.grad


class TestGradChannel(TestGrad):
    def __init__(self, n=3):
        self.params = Parameter(torch.Tensor([1.0] * 4))
        a, b, c, d = self.params
        B1 = Block(n, a, b)
        B2 = Block(n, c, d)
        self.circuit = Lindblad(B1, B2)
        self.psi0 = zero_state(n)
        self.target = zero_state(n)

    def optimal_control_grad(self, randomness):
        def Obs(y):
            return self.target * cdot(self.target, y)

        self.circuit.optimal_control(self.psi0, Obs, randomness=randomness)
        return self.params.grad


def average(get_grad, nparams, samples):
    from tqdm import tqdm

    randomness = np.random.uniform(0, 1, (samples, nparams))
    sum = 0
    # for i, rand in tqdm(enumerate(randomness)):
    for i, rand in enumerate(randomness):
        sample = get_grad(randomness=rand)
        sum += sample
        if i % 100 == 0:
            print(sum / (i + 1))

    return sum / samples


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=4)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    print(TestGrad().optimal_control_grad())
    print(TestGrad().auto_grad())
    print(TestGrad().density_matrix_grad())

    print(TestGradChannel().density_matrix_grad())

    def estimate(randomness):
        return TestGradChannel().optimal_control_grad(randomness=randomness)

    print(average(estimate, 4, 1000))

    # print("\nquantum control gradient  ", test_grad_unitary.get_grad())
    # print("\nsanity check (param shift)", test_grad_unitary.get_grad_paramshift())

    # test_grad_channel = TestGradChannel()
    # print("\nsampling channel for quantum control gradient")
    # print(average(test_grad_channel.get_grad, 1000))
    # print("\nsampling channel for param shift sanity check")
    # print(average(test_grad_channel.get_grad_paramshift, 1000))
