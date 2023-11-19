import argparse
from test import TestGradChannel
from examples import *
from torch import optim
import os
from differentiable_circuit import *
from datatypes import *
from torch.nn import Parameter, ParameterList
import torch
import examples
import pickle
import json
from torch import nn


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
    argparser.add_argument("--n", type=int, default=6)
    argparser.add_argument("--l", type=int, default=6)
    argparser.add_argument("--d", type=int, default=1)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--iterations_per_epoch", type=int, default=1000)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    d = args.d
    outdir = makedir(args.outdir)

    def emph(txt):
        print(f"\n{80 * '_'}\n\n{txt}\n{80*'_'}\n")

    emph(f"{n}+1 qubits")

    def groundstate(H: Hamiltonian, n: int):
        H = H.create_dense(n)
        energies, states = torch.linalg.eigh(H)
        return states[:, 0]

    H = TFIM(n)
    # reverse_block = UnitaryBlock(H, l=l, reverse=True)

    block = UnitaryBlock(H, l=l)
    reverse_block = block.get_reverse()

    optimizer = optim.Adam(reverse_block.parameters(), lr=0.01)
    target = groundstate(H, n)
    rho_target = target[:, None] * target[None, :].conj()
    add_random_ancilla = AddRandomAncilla(p=0)
    restrict = RestrictMeasurementOutcome(p=0)

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(H, f"{outdir}/H.pt")

    blocks = []

    for epoch in range(args.epochs):
        torch.save(optimizer.state_dict(), f"{outdir}/optimizer_{epoch}.pt")
        torch.save(rho_target, f"{outdir}/rho_t-{epoch}.pt")
        circuit = CircuitChannel(gates=blocks)
        torch.save(circuit, f"{outdir}/circuit_{epoch}.pt")
        torch.save(circuit.state_dict(), f"{outdir}/circuit_params_{epoch}.pt")

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            rho_out = add_random_ancilla.apply_to_density_matrix(rho_target)
            rho_in = reverse_block.apply_to_density_matrix(rho_out)
            rho_in_restricted = restrict.apply_to_density_matrix(rho_in)
            value = rho_in_restricted.trace().real

            loss = -value
            loss.backward()

            # breakpoint()
            optimizer.step()

            print(value)

            with open(f"{outdir}/values.txt", "a") as f:
                f.write(f"{epoch} {i} {value.detach().cpu().numpy()}\n")

        rho_target = (rho_in_restricted / rho_in_restricted.trace().real).detach()
        emph(f"After {epoch+1} epochs: {value:.5f}")

        forward_block = reverse_block.get_reverse(mode="measurement")
        torch.save(forward_block, f"{outdir}/block_t-{epoch}.pt")
        torch.save(forward_block.state_dict(), f"{outdir}/params_t-{epoch}.pt")

        blocks.append(forward_block)
        reverse_block = copy.deepcopy(reverse_block)
