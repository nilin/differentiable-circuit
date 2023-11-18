import config
import numpy as np
import torch
import argparse
from differentiable_circuit import cdot, squared_overlap, CircuitChannel
from differentiable_gate import *
from examples import Block, TFIM, HaarState, ZeroState
import examples
from torch.nn import Parameter, ParameterList
from datatypes import *


class TestGradChannel:
    def __init__(self, n=6):
        self.n = n

        # zetas1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        # taus1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        # self.params = ParameterList(zetas1 + taus1)
        # self.circuit = Block(self.n, taus1, zetas1)

        as1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        as2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        as3 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas3 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus3 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        self.params = ParameterList(
            as1 + as2 + as3 + taus1 + zetas1 + taus2 + zetas2 + taus3 + zetas3
        )

        self.H = TFIM(self.n)
        B1 = Block(self.H, as1, taus1, zetas1)
        B2 = Block(self.H, as2, taus2, zetas2)
        B3 = Block(self.H, as3, taus3, zetas3)
        self.circuit = CircuitChannel(gates=[B1, B2, B3])

        self.prepstates()

    def prepstates(self):
        self.psi0 = examples.ZeroState(self.n).pure_state()
        self.target = examples.HaarState(
            self.n, torch.Generator(device=config.device)
        ).pure_state()
        # self.target = self.groundstate()
        # self.target = zero_state(self.n + 1)
        self.Obs = lambda y: self.target * cdot(self.target, y)

    def groundstate(self):
        H = self.H.create_dense(self.n)
        energies, states = torch.linalg.eigh(H)
        return states[:, 0]

    def optimal_control_grad(self, **kwargs):
        value, _ = self.circuit.optimal_control(self.psi0, self.Obs, **kwargs)
        return self.reformat(value, torch.autograd.grad(value, self.params))

    def autograd(self):
        value = squared_overlap(self.target, self.circuit.apply(self.psi0))
        return self.reformat(value, torch.autograd.grad(value, self.params))

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()
        # rho = self.psi0.density_matrix()
        rho_out = self.circuit.apply_to_density_matrix(rho)

        value = cdot(self.target, rho_out @ self.target).real
        return self.reformat(value, torch.autograd.grad(value, self.params))

    def paramshift_grad(self, e=0.001, **kwargs):
        value = squared_overlap(self.target, self.circuit.apply(self.psi0, **kwargs))
        grad = []
        for i, p in enumerate(self.params):
            with torch.no_grad():
                curval = p.item()
                p.copy_(p + e)
                loss_p = squared_overlap(self.target, self.circuit.apply(self.psi0, **kwargs))
                p.copy_(curval)
            grad.append(((loss_p - value) / e).cpu())
        return self.reformat(value, grad)

    @staticmethod
    def reformat(value, grad):
        return value.cpu().detach(), np.stack([g.detach().numpy() for g in grad])


class TestGradUnitary(TestGradChannel):
    def __init__(self, n=6):
        self.n = n
        a_ = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        self.params = ParameterList(zetas + taus)

        self.H = TFIM(self.n)
        self.circuit = Block(self.H, a_, taus, zetas)

        """Remove non-unitary operations"""
        self.circuit.gates = self.circuit.gates[1:-1]

        self.prepstates()

    def prepstates(self):
        self.psi0 = examples.ZeroState(self.n + 1).pure_state()
        self.target = examples.HaarState(
            self.n + 1, torch.Generator(device=config.device)
        ).pure_state()
        # self.target = self.groundstate()
        # self.target = zero_state(self.n + 1)
        self.Obs = lambda y: self.target * cdot(self.target, y)

    def groundstate(self):
        H = self.H.create_dense(self.n)
        energies, states = torch.linalg.eigh(H)
        return torch.cat([states[:, 0], torch.zeros_like(states[:, 0])])


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
        f"\n{txt}\nsigned overlap in [-1,1]: {np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)):.5f}"
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n", type=int, default=(8 if config.device.type == "cpu" else 9))
    args, _ = argparser.parse_known_args()
    n = args.n

    def EMPH(txt):
        print()
        print(80 * "_" + "\n")
        print(txt)
        print(80 * "_" + "\n")

    EMPH(f"{n}+1 qubits")

    np.random.seed(0)
    torch.manual_seed(1)

    print("preparing problems")
    testgrad = TestGradUnitary(n)
    testgradchannel = TestGradChannel(n)
    print("preparing reference values using density matrix autograd")
    ref_unitary = testgrad.density_matrix_grad()
    ref_channel = testgradchannel.density_matrix_grad()
    print("reference methods done")

    EMPH("Compute gradient of unitary circuit. Overlap with reference gradient")

    compare(ref_unitary, testgrad.optimal_control_grad(), "method: optimal control grad")
    compare(ref_unitary, testgrad.autograd(), "method: autograd")
    # compare(ref_unitary, testgrad.paramshift_grad(), "method: param shift")

    EMPH(
        "Estimate gradient of channel using [optimal control].\n\n"
        + "Overlap with true gradient computed using density matrix evolution"
    )

    def estimate(randomness):
        return testgradchannel.optimal_control_grad(randomness=randomness)

    for i, data in sample(estimate, 4, checkpoint_times=[1, 10, 50, 100]):
        value = np.stack(list(zip(*data))[0]).mean()
        grad = np.stack(list(zip(*data))[1]).mean(axis=0)
        compare(ref_channel, (value, grad), f"{i} samples")

    # EMPH(
    #    "Estimate gradient of channel using [parameter shift].\n\n"
    #    + "Overlap with true gradient computed using density matrix evolution"
    # )

    # def estimate(randomness):
    #    return testgradchannel.paramshift_grad(randomness=randomness)

    # for i, data in sample(estimate, 4, checkpoint_times=[1, 10, 50, 100]):
    #    value = np.stack(list(zip(*data))[0]).mean()
    #    grad = np.stack(list(zip(*data))[1]).mean(axis=0)
    #    compare(ref_channel, (value, grad), f"{i}x{len(grad)} passes")

    # EMPH("done")
