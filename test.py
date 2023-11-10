from config import device
import config
import numpy as np
import torch
import argparse
from differentiable_circuit import Params, overlap, Circuit
from typing import Literal
from typing import List, Callable
import examples


class TestGrad:
    shiftparam = 0.001
    L: int
    test_problem: Callable
    params: List[float]

    def get_grad(self, **kw):
        a, b, c, d = Params.def_param(*self.params)
        C, x, target = self.test_problem(a, b, c, d)

        def Obs(y):
            return target * overlap(target, y)

        C.optimal_control(x, Obs, **kw)
        return torch.Tensor([p.grad for p in [a, b, c, d]])

    def get_grad_paramshift(self, **kw):
        a, b, c, d = Params.def_param(*self.params)
        s = self.shiftparam

        def getloss(*params):
            C, x, target = self.test_problem(*params)
            y = C.apply(x, **kw)
            return torch.abs(overlap(target, y)) ** 2

        def estimate(*s_params):
            return (getloss(*s_params) - getloss(a, b, c, d)).real / s

        a, b, c, d = self.params
        return torch.Tensor(
            (
                estimate(a + s, b, c, d),
                estimate(a, b + s, c, d),
                estimate(a, b, c + s, d),
                estimate(a, b, c, d + s),
            )
        )


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
        B1 = examples.Block(L, a, b)
        B2 = examples.Block(L, c, d)
        C = Circuit(gates=B1.gates + B2.gates)
        x = examples.zero_state(L)
        target = examples.zero_state(L)
        return C, x, target


class TestGradChannel(TestGrad):
    shiftparam = 0.1
    L = 3
    params = [1.0] * 4

    def test_problem(self, a, b, c, d):
        L = self.L
        B1 = examples.Block(L, a, b)
        B2 = examples.Block(L, c, d)
        C = examples.Lindblad(B1, B2)
        x = examples.zero_state(L)
        target = examples.zero_state(L)
        return C, x, target


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=4)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    test_grad_unitary = TestGradUnitary()
    print("\nquantum control gradient  ", test_grad_unitary.get_grad())
    print("\nsanity check (param shift)", test_grad_unitary.get_grad_paramshift())

    test_grad_channel = TestGradChannel()
    print("\nsampling channel for quantum control gradient")
    print(average(test_grad_channel.get_grad, 1000))
    print("\nsampling channel for param shift sanity check")
    print(average(test_grad_channel.get_grad_paramshift, 1000))
