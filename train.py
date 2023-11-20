import argparse
from test import TestGradChannel
from examples import *
from non_unitary_gates import *
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


def emph(txt):
    print(f"\n{80 * '_'}\n\n{txt}\n{80*'_'}\n")


def groundstate(H: Hamiltonian, n: int):
    H = H.create_dense(n)
    energies, states = torch.linalg.eigh(H)
    return states[:, 0]


def testcircuit(circuit, target):
    I = torch.eye(len(target), dtype=tcomplex, device=config.device) / len(target)
    rho_t = circuit.apply_to_density_matrix(I)
    value = cdot(target, rho_t @ target).real
    # print(f"Circuit value: {value:.5f}")
    return value


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n", type=int, default=6)
    argparser.add_argument("--l", type=int, default=6)
    argparser.add_argument("--d", type=int, default=1)
    argparser.add_argument("--trottersteps", type=int, default=1)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--iterations_per_epoch", type=int, default=1000)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    d = args.d
    outdir = makedir(args.outdir)

    emph(f"{n}+1 qubits")

    H = TFIM(n)
    target = groundstate(H, n)
    rho_target = target[:, None] * target[None, :].conj()

    add_random_ancilla = AddRandomAncilla(0)
    restrict = RestrictMeasurementOutcome(0)

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(H, f"{outdir}/H.pt")

    blocks = []

    for epoch in range(args.epochs):
        block = UnitaryBlock(H, l=l, trottersteps=args.trottersteps)
        reverse_block = block.get_reverse()
        optimizer = optim.Adam(reverse_block.parameters(), lr=0.01)

        # torch.save(optimizer.state_dict(), f"{outdir}/optimizer_{epoch}.pt")
        torch.save(rho_target, f"{outdir}/rho_t-{epoch}.pt")

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            rho_out = add_random_ancilla.apply_to_density_matrix(rho_target)
            rho_in = reverse_block.apply_to_density_matrix(rho_out)
            rho_in_restricted = restrict.apply_to_density_matrix(rho_in)
            value = rho_in_restricted.trace().real

            loss = -value
            loss.backward()

            optimizer.step()

            print(value)

            with open(f"{outdir}/values.txt", "a") as f:
                f.write(f"{epoch} {i} {value.detach().cpu().numpy()}\n")

        rho_target = (rho_in_restricted / rho_in_restricted.trace().real).detach()

        forward_block = reverse_block.get_reverse(mode="measurement")
        torch.save(forward_block, f"{outdir}/block_t-{epoch}.pt")
        torch.save(forward_block.state_dict(), f"{outdir}/params_t-{epoch}.pt")

        blocks.append(Block(unitaryblock=forward_block))

        circuit = CircuitChannel(gates=blocks)
        torch.save(circuit, f"{outdir}/circuit_{epoch+1}.pt")
        torch.save(circuit.state_dict(), f"{outdir}/circuit_params_{epoch}.pt")

        circuitvalue = testcircuit(circuit, target)
        emph(f"After {epoch+1} epochs: circuit value {circuitvalue:.5f}")
