import config
import numpy as np
import torch
import argparse
from differentiable_circuit import cdot, squared_overlap, Channel
from differentiable_gate import *
from examples import Block, zero_state
from torch.nn import Parameter, ParameterList
from datatypes import *


def retrieve(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
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

        value, _ = self.circuit.optimal_control(self.psi0, Obs, **kwargs)
        return self.reformat(value, torch.autograd.grad(value, self.params))

    def auto_grad(self):
        value = squared_overlap(self.target, self.circuit.apply(self.psi0))
        return self.reformat(value, torch.autograd.grad(value, self.params))

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()
        rho_out = self.circuit.apply_to_density_matrix(rho)

        value = cdot(self.target, rho_out @ self.target).real
        return self.reformat(value, torch.autograd.grad(value, self.params))

    def paramshift_grad(self, e=0.01, **kwargs):
        value = squared_overlap(self.target, self.circuit.apply(self.psi0, **kwargs))
        grad = []
        for i, p in enumerate(self.params):
            with torch.no_grad():
                curval = p.item()
                p.copy_(p + e)
                loss_p = squared_overlap(
                    self.target, self.circuit.apply(self.psi0, **kwargs)
                )
                p.copy_(curval)
            grad.append(((loss_p - value) / e).cpu())
        return self.reformat(value, grad)

    @staticmethod
    def reformat(value, grad):
        return value.cpu().detach(), np.stack([g.detach().numpy() for g in grad])


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

        value, _ = self.circuit.optimal_control(self.psi0, Obs, randomness=randomness)
        return self.reformat(value, torch.autograd.grad(value, self.params))


def sample(get_grad, nparams, checkpoint_times):
    samples = checkpoint_times[-1]
    randomness = np.random.uniform(0, 1, (samples, nparams))
    data = []

    for i, rand in enumerate(randomness):
        data.append(get_grad(randomness=rand))
        if (i + 1) in checkpoint_times:
            yield i + 1, data


def compare(a, b, txt):
    a, b = a[1], b[1]
    print(
        f"\n{txt}\nsigned overlap in [-1,1]: {np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))}"
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--n", type=int, default=(10 if config.device.type == "cpu" else 10)
    )
    args, _ = argparser.parse_known_args()
    n = args.n

    print(f"\n{n} qubits")

    print("\ntest unitary circuit")
    testgrad = TestGrad(n)
    print("reference method: autograd")
    ref = testgrad.auto_grad()
    compare(ref, testgrad.optimal_control_grad(), "method: optimal control grad")
    compare(ref, testgrad.density_matrix_grad(), "method: density matrix autograd")
    compare(ref, testgrad.paramshift_grad(), "method: param shift")

    print("\nTest channel")

    testgradchannel = TestGradChannel(n)
    print("reference method: autograd for density matrix")
    ref = testgradchannel.density_matrix_grad()

    print("\nMethod: Quantum control for channels")

    def estimate(randomness):
        return testgradchannel.optimal_control_grad(randomness=randomness)

    for i, data in sample(estimate, 4, checkpoint_times=[5, 10]):
        value = np.stack(list(zip(*data))[0]).mean()
        grad = np.stack(list(zip(*data))[1]).mean(axis=0)
        compare(ref, (value, grad), f"quantum control with {i} samples")

    print("\nMethod: param shift")

    def estimate(randomness):
        return testgradchannel.paramshift_grad(randomness=randomness)

    for i, data in sample(estimate, 4, checkpoint_times=[5, 10]):
        value = np.stack(list(zip(*data))[0]).mean()
        grad = np.stack(list(zip(*data))[1]).mean(axis=0)
        compare(ref, (value, grad), f"parameter shift with {i}x{len(grad)} passes")
