from config import device
import config
import numpy as np
import torch
import argparse
from differentiable_circuit import cdot, Circuit, squared_overlap, Channel
from differentiable_gate import *
from typing import Literal
from typing import List, Callable
import examples
from functools import partial
from examples import Block, zero_state
import copy
from gate_implementation import TorchGate, EvolveDensityMatrix, GateImplementation
from torch.nn import Parameter, ParameterList
from torch.utils import _pytree
from scipy.stats import bootstrap
from datatypes import *


def retrieve(x):
    return x.cpu().detach().numpy().round(4)


class TestGrad:
    def __init__(self, n=6):
        self.n = n
        zetas = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        self.params = ParameterList(zetas + taus)
        self.circuit = Block(self.n, taus, zetas, with_reset=False)

        self.psi0 = zero_state(self.n)
        self.target = zero_state(self.n)

    def optimal_control_grad(self, **kwargs):
        def Obs(y):
            return self.target * cdot(self.target, y)

        expectation, _ = self.circuit.optimal_control(self.psi0, Obs, **kwargs)
        return self.format(expectation)

    def auto_grad(self):
        loss = squared_overlap(self.target, self.circuit.apply(self.psi0))
        loss.backward()
        return self.format(loss)

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()
        rho_out = self.circuit.apply_to_density_matrix(rho)

        loss = cdot(self.target, rho_out @ self.target).real
        loss.backward()
        return self.format(loss)

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
            grad.append(((loss_p - loss) / e))
        return self.format(loss, grad)

    def format(self, loss, grads=None):
        if grads is None:
            grads = [retrieve(p.grad) for p in self.params]
        else:
            grads = [retrieve(g) for g in grads]
        return {"loss": retrieve(loss), "grad": grads}


class TestGradChannel(TestGrad):
    def __init__(self, n=6):
        self.n = n

        # zetas1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        # taus1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        # self.params = ParameterList(zetas1 + taus1)
        # self.circuit = Block(self.n, taus1, zetas1)

        zetas1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        self.params = ParameterList(zetas1 + taus1 + zetas2 + taus2)

        B1 = Block(self.n, taus1, zetas1)
        B2 = Block(self.n, taus2, zetas2)
        self.circuit = Channel(B1.gates + B2.gates)

        self.psi0 = zero_state(self.n)
        self.target = zero_state(self.n)

    def optimal_control_grad(self, randomness):
        def Obs(y):
            return self.target * cdot(self.target, y)

        expectation, _ = self.circuit.optimal_control(
            self.psi0, Obs, randomness=randomness
        )
        return self.format(expectation)


def sample(get_grad, print_stats, nparams, samples):
    randomness = np.random.uniform(0, 1, (samples, nparams))
    data = []

    for i, rand in enumerate(randomness):
        data.append(get_grad(randomness=rand))
        if (i + 1) in [10, 100, 500, 1000, samples]:
            print_stats(data)

    return data


def print_stats(data):
    shape = _pytree.tree_flatten(data[0])[1]
    table = [_pytree.tree_flatten(datapt)[0] for datapt in data]
    columns = list(zip(*table))

    bootstraps = [
        bootstrap((np.array(col),), np.mean, n_resamples=1000, confidence_level=0.99)
        for col in columns
    ]
    low = [b.confidence_interval.low.round(4) for b in bootstraps]
    high = [b.confidence_interval.high.round(4) for b in bootstraps]

    print(f"\n99% Confidence intervals (high/low: top/bottom), {len(data)} samples")
    print(_pytree.tree_unflatten(high, shape))
    print(_pytree.tree_unflatten(low, shape))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n", type=int, default=4)
    args, _ = argparser.parse_known_args()
    n = args.n

    # np.set_printoptions(precision=3, suppress=True)

    print("\ntest unitary circuit")
    print(TestGrad(n).optimal_control_grad())
    print(TestGrad(n).auto_grad())
    print(TestGrad(n).density_matrix_grad())
    print(TestGrad(n).paramshift_grad())

    print("\ntest channel")

    print("density matrix")
    print(TestGradChannel(n).density_matrix_grad())

    print("\nquantum control")

    def estimate(randomness):
        return TestGradChannel(n).optimal_control_grad(randomness=randomness)

    sample(estimate, print_stats, 4, 1000)

    print("\nparam shift")

    def estimate(randomness):
        return TestGradChannel(n).paramshift_grad(randomness=randomness)

    sample(estimate, print_stats, 4, 1000)
