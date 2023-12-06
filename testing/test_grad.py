import config
import numpy as np
import torch
import argparse
from gate import *
from examples import Block, TFIM, HaarState, ZeroState, UnitaryBlock
import examples
from torch.nn import Parameter, ParameterList
from datatypes import *


class Circuit(Non_unitary_circuit):
    def __init__(self, l, d, H):
        nn.Module.__init__(self)
        self.gates = nn.ModuleList([Block(H, l=l) for _ in range(d)])


class UnitaryCircuit(Non_unitary_circuit):
    def __init__(self, l, d, H):
        nn.Module.__init__(self)
        self.gates = nn.ModuleList([UnitaryBlock(H, l=l) for _ in range(d)])


class TestGradChannel:
    def __init__(self, n=6):
        self.n = n
        self.circuit = Circuit(3, 3, TFIM(self.n))
        self.prepstates()

    def prepstates(self):
        self.psi0 = examples.ZeroState(self.n).pure_state()
        self.target = examples.HaarState(
            self.n, torch.Generator(device=config.device)
        ).pure_state()
        self.Obs = lambda y: torch.abs(cdot(self.target, y) ** 2)

    def groundstate(self):
        H = self.H.create_dense_matrix(self.n)
        energies, states = torch.linalg.eigh(H)
        return states[:, 0]

    def optimal_control_grad(self, **kwargs):
        value, *_ = self.circuit.optimal_control(self.psi0, self.Obs, **kwargs)
        return self.reformat(value, torch.autograd.grad(value, self.circuit.parameters()))

    def autograd(self):
        value = squared_overlap(self.target, self.circuit.apply(self.psi0))
        return self.reformat(value, torch.autograd.grad(value, self.circuit.parameters()))

    def density_matrix_grad(self):
        rho = self.psi0[:, None] * self.psi0[None, :].conj()
        rho_out = self.circuit.apply_to_density_matrix(rho)

        value = cdot(self.target, rho_out @ self.target).real
        return self.reformat(value, torch.autograd.grad(value, self.circuit.parameters()))

    @staticmethod
    def reformat(value, grad):
        return value.cpu().detach(), np.stack([g.detach().cpu().numpy() for g in grad])


class TestGradUnitary(TestGradChannel):
    def __init__(self, n=6):
        self.n = n
        self.circuit = UnitaryCircuit(3, 3, TFIM(self.n))
        self.prepstates()

    def prepstates(self):
        self.psi0 = examples.ZeroState(self.n + 1).pure_state()
        self.target = examples.HaarState(
            self.n + 1, torch.Generator(device=config.device)
        ).pure_state()
        self.Obs = lambda y: torch.abs(cdot(self.target, y) ** 2)

    def groundstate(self):
        H = self.H.create_dense_matrix(self.n)
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
