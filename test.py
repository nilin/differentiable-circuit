from config import device
import config
import numpy as np
import torch
import argparse
from differentiable_circuit import cdot, Circuit, squared_overlap, Channel
from differentiable_gate import NoMeasurement, CleanSlateAncilla
from typing import Literal
from typing import List, Callable
import examples
from functools import partial
from examples import Block, Lindblad, zero_state
import copy
from gate_implementation import TorchGate, EvolveDensityMatrix, GateImplementation
from torch.nn import Parameter, ParameterList


class TestGrad:
    def __init__(self, n=4):
        self.n = n
        initial_params = [1.0] * 4
        self.params = ParameterList([Parameter(torch.tensor(p)) for p in initial_params])
        a, b, c, d = self.params
        B1 = Block(self.n, a, b)
        B2 = Block(self.n, c, d)
        self.circuit = Circuit(gates=B1.gates + B2.gates)
        self.psi0 = zero_state(self.n)
        self.target = zero_state(self.n)

    def optimal_control_grad(self, **kwargs):
        def Obs(y):
            return self.target * cdot(self.target, y)

        expectation, _ = self.circuit.optimal_control(self.psi0, Obs, **kwargs)
        return expectation.cpu(), [p.grad for p in self.params]

    def auto_grad(self):
        loss = squared_overlap(self.target, self.circuit.apply(self.psi0))
        loss.backward()
        return loss.cpu(), [p.grad for p in self.params]

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()
        rho_out = self.circuit.apply_to_density_matrix(rho)

        loss = cdot(self.target, rho_out @ self.target).real
        loss.backward()
        return loss.cpu(), [p.grad for p in self.params]

    def paramshift_grad(self, e=0.01, **kwargs):
        loss = squared_overlap(self.target, self.circuit.apply(self.psi0, **kwargs))
        grad = []
        for i, p in enumerate(self.params):
            with torch.no_grad():
                curval = p.item()
                p.copy_(p + e)
                loss_p = squared_overlap(
                    self.target, self.circuit.apply(self.psi0, **kwargs)
                )
                p.copy_(curval)
            grad.append(((loss_p - loss) / e).cpu().detach())
        return loss.cpu(), grad


class TestGradChannel(TestGrad):
    def __init__(self, n=3):
        self.n = n
        initial_params = [1.0] * 4
        self.params = ParameterList([Parameter(torch.tensor(p)) for p in initial_params])
        a, b, c, d = self.params
        B1 = Block(self.n, a, b)
        B2 = Block(self.n, c, d)

        self.circuit = Lindblad(B1, B2)
        self.psi0 = zero_state(self.n)
        self.target = zero_state(self.n)

        # self.circuit = Channel(blocks=[B1, B2], measurements=[NoMeasurement()] * 2)
        # self.psi0 = zero_state(n)
        # self.target = zero_state(n)

        # self.params = Parameter(torch.Tensor([1.0] * 2))
        # a, b = self.params
        # B1 = Block(n, a, b)
        # self.circuit = Channel(blocks=[B1], measurements=[CleanSlateAncilla(0)])
        # self.psi0 = zero_state(n)
        # self.target = zero_state(n)

    def optimal_control_grad(self, randomness):
        def Obs(y):
            return self.target * cdot(self.target, y)

        expectation, _ = self.circuit.optimal_control(
            self.psi0, Obs, randomness=randomness
        )
        return expectation.cpu(), [p.grad for p in self.params]


def average(get_grad, nparams, samples):
    from tqdm import tqdm

    randomness = np.random.uniform(0, 1, (samples, nparams))
    sum_e = 0
    sum_grad = [0] * nparams
    # for i, rand in tqdm(enumerate(randomness)):
    for i, rand in enumerate(randomness):
        e, grad = get_grad(randomness=rand)
        sum_e += e
        sum_grad = [s + g for s, g in zip(sum_grad, grad)]
        if i % 100 == 0:
            print(sum_e / (i + 1), [s / (i + 1) for s in sum_grad])

    return sum_e / samples, [s / samples for s in sum_grad]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=4)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    print("\ntest unitary circuit")
    print(TestGrad().optimal_control_grad())
    print(TestGrad().auto_grad())
    print(TestGrad().density_matrix_grad())
    print(TestGrad().paramshift_grad())

    print("\ntest channel")
    print(TestGradChannel().density_matrix_grad())

    print("\nquantum control")

    def estimate(randomness):
        return TestGradChannel().optimal_control_grad(randomness=randomness)

    print(average(estimate, 4, 1000))

    print("\nparam shift")

    def estimate(randomness):
        return TestGradChannel().paramshift_grad(randomness=randomness)

    print(average(estimate, 4, 1000))

    # print("\nquantum control gradient  ", test_grad_unitary.get_grad())
    # print("\nsanity check (param shift)", test_grad_unitary.get_grad_paramshift())

    # test_grad_channel = TestGradChannel()
    # print("\nsampling channel for quantum control gradient")
    # print(average(test_grad_channel.get_grad, 1000))
    # print("\nsampling channel for param shift sanity check")
    # print(average(test_grad_channel.get_grad_paramshift, 1000))
