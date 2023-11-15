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


class Problem:
    H: Hamiltonian
    target: State
    initial_state: StateGenerator
    seed: int = 0

    def __init__(self, n):
        self.n = n
        self.H = TFIM(self.n)
        self.target = self.groundstate(self.H, self.n)
        self.Obs = lambda y: self.target * cdot(self.target, y)

    def preptarget(self):
        self.target = self.groundstate()
        self.Obs = lambda y: self.target * cdot(self.target, y)

    @staticmethod
    def groundstate(H: Hamiltonian, n: int):
        H = H.create_dense(n)
        energies, states = torch.linalg.eigh(H)
        return states[:, 0]


def getcircuit(n, l, d, problem):
    _a_ = [[Parameter(torch.tensor(1.0)) for _ in range(l)] for _ in range(d)]
    _tau_ = [[Parameter(torch.tensor(1.0)) for _ in range(l)] for _ in range(d)]
    _zeta_ = [[Parameter(torch.tensor(1.0)) for _ in range(l)] for _ in range(d)]
    params = ParameterList(
        [a for as_ in _a_ for a in as_]
        + [tau for taus_ in _tau_ for tau in taus_]
        + [zeta for zetas_ in _zeta_ for zeta in zetas_]
    )

    circuit = CircuitChannel(
        gates=[
            Block(problem.H, a_, tau_, zeta_, mixwith=list(range(1, n)) * 10)
            for a_, tau_, zeta_ in zip(_a_, _tau_, _zeta_)
        ]
    )
    return circuit, params

    # a = Parameter(torch.tensor(1.0))
    # tau = Parameter(torch.tensor(1.0))
    # zeta = Parameter(torch.tensor(1.0))
    # params = ParameterList([a, tau, zeta])
    # return ShortBlock(problem.H, a, tau, zeta), params


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
    block, params = getcircuit(n, l, d, problem)

    optimizer = optim.Adam(params, lr=0.01)
    values = []

    rho = ZeroState(n).density_matrix()

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(problem.H, f"{outdir}/H.pt")

    for epoch in range(args.epochs):
        problem.rho = rho

        torch.save(optimizer.state_dict(), f"{outdir}/optimizer_{epoch}.pt")
        torch.save(rho, f"{outdir}/rho_{epoch}.pt")
        torch.save(block, f"{outdir}/block_{epoch}.pt")
        torch.save(params, f"{outdir}/params_{epoch}.pt")

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            rho_out = block.apply_to_density_matrix(rho)
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
