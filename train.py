import argparse
from test import TestGradChannel
from examples import *
from torch import optim
import os
from datatypes import *
from torch.nn import Parameter, ParameterList
import torch
import examples
import pickle
import json
from torch import nn


class Problem:
    H: Hamiltonian
    target: State
    initial_state: StateGenerator
    seed: int = 0

    def __init__(self, n):
        self.n = n
        self.H = TFIM(self.n)
        self.preptarget()
        self.Obs = lambda y: self.target * cdot(self.target, y)

    def preptarget(self):
        gs = self.groundstate(self.H, self.n)
        self.target = torch.cat([gs, torch.zeros_like(gs)])
        self.Obs = lambda y: self.target * cdot(self.target, y)

    @staticmethod
    def groundstate(H: Hamiltonian, n: int):
        H = H.create_dense(n)
        energies, states = torch.linalg.eigh(H)
        return states[:, 0]


class Circuit(CircuitChannel):
    def __init__(self, l, d, H):
        nn.Module.__init__(self)
        self.gates = nn.ModuleList([UnitaryBlock(H, l=l) for _ in range(d)])


def makedir(path):
    rpath = path
    i = 1
    while os.path.exists(path):
        path = f"{rpath}_{i}"
        i += 1
    os.makedirs(path)
    return path


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n", type=int, default=8)
    argparser.add_argument("--l", type=int, default=40)
    argparser.add_argument("--d", type=int, default=1)
    argparser.add_argument("--epochs", type=int, default=10000)
    argparser.add_argument("--iterations_per_epoch", type=int, default=10)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    d = args.d
    outdir = makedir(args.outdir)

    def emph(txt):
        print(f"\n{80 * '_'}\n\n{txt}\n{80*'_'}\n")

    emph(f"{n}+1 qubits")

    problem = Problem(n)
    circuit = Circuit(l, d, problem.H)
    # circuit = circuit.reverse()

    optimizer = optim.Adam(circuit.parameters(), lr=0.01)
    values = []

    rho = ZeroState(n + 1).density_matrix()

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(problem.H, f"{outdir}/H.pt")

    for epoch in range(args.epochs):
        problem.rho = rho

        torch.save(optimizer.state_dict(), f"{outdir}/optimizer_{epoch}.pt")
        torch.save(rho, f"{outdir}/rho_{epoch}.pt")
        torch.save(circuit, f"{outdir}/block_{epoch}.pt")
        torch.save(circuit.state_dict(), f"{outdir}/params_{epoch}.pt")

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            rho_out = circuit.apply_to_density_matrix(rho)
            value = cdot(problem.target, rho_out @ problem.target).real

            loss = -value
            loss.backward()
            optimizer.step()

            print(value)
            values.append(value.detach().cpu().numpy())

            with open(f"{outdir}/values.txt", "a") as f:
                f.write(f"{epoch} {i} {value.detach().cpu().numpy()}\n")

        rho = rho_out.detach()
        emph(f"After {epoch+1} epochs: {value:.5f}")

    emph("Final state:")
