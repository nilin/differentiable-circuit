from config import device
import config
import numpy as np
import torch
import argparse
from differentiable_circuit import Params, cdot, Circuit, squared_overlap
from typing import Literal
from typing import List, Callable
import gates_and_circuits


class TestGrad:
    shiftparam = 0.001
    L: int
    test_problem: Callable
    params: List[float]

    def get_grad(self, **kw):
        a, b, c, d = Params().def_param(*self.params)
        C, x, target = self.test_problem(a, b, c, d)

        def Obs(y):
            return target * cdot(target, y)

        C.optimal_control(x, Obs, **kw)
        return torch.Tensor([p.grad for p in [a, b, c, d]])

    def autograd(self, **kw):
        a, b, c, d = Params().def_param(*self.params)
        C, x, target = self.test_problem(a, b, c, d)

        loss = squared_overlap(target, C.apply(x, **kw))
        loss.backward()
        return torch.Tensor([p.grad for p in [a, b, c, d]])


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


class TestGradUnitary(TestGrad):
    L = 4
    params = [1.0] * 4

    def test_problem(self, a, b, c, d):
        L = self.L
        B1 = gates_and_circuits.Block(L, a, b)
        B2 = gates_and_circuits.Block(L, c, d)
        C = Circuit(gates=B1.gates + B2.gates)
        x = gates_and_circuits.zero_state(L)
        target = gates_and_circuits.zero_state(L)
        return C, x, target


class TestGradChannel(TestGrad):
    shiftparam = 0.1
    L = 3
    params = [1.0] * 4

    def test_problem(self, a, b, c, d):
        L = self.L
        B1 = gates_and_circuits.Block(L, a, b)
        B2 = gates_and_circuits.Block(L, c, d)
        C = gates_and_circuits.Lindblad(B1, B2)
        x = gates_and_circuits.zero_state(L)
        target = gates_and_circuits.zero_state(L)
        return C, x, target


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=4)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    test_grad_unitary = TestGradUnitary()
    print(test_grad_unitary.get_grad())
    print(test_grad_unitary.autograd())

    # print("\nquantum control gradient  ", test_grad_unitary.get_grad())
    # print("\nsanity check (param shift)", test_grad_unitary.get_grad_paramshift())

    # test_grad_channel = TestGradChannel()
    # print("\nsampling channel for quantum control gradient")
    # print(average(test_grad_channel.get_grad, 1000))
    # print("\nsampling channel for param shift sanity check")
    # print(average(test_grad_channel.get_grad_paramshift, 1000))
