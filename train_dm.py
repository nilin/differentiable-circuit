import argparse
from test_grad import TestGradChannel
from examples import *
from non_unitary_gates import *
from torch import optim
import os
from differentiable_channel import *
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
    H = H.create_dense_matrix(n)
    energies, states = torch.linalg.eigh(H)
    return states[:, 0]


def testcircuit(circuit, target):
    I = torch.eye(len(target), dtype=tcomplex, device=config.device) / len(target)
    I = I.detach()
    rho_t = circuit.apply_to_density_matrix(I)
    value = cdot(target, rho_t @ target).real
    # print(f"Circuit value: {value:.5f}")
    return value


def retrieve(*values):
    for value in values:
        yield value.detach().cpu().numpy()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dm", action="store_true")
    argparser.add_argument("--n", type=int, default=8)
    argparser.add_argument("--l", type=int, default=12)
    argparser.add_argument("--trottersteps", type=int, default=1)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--iterations_per_epoch", type=int, default=5000)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    argparser.add_argument("--reload", type=str, default="")
    argparser.add_argument("--compare_args", type=str, default="")

    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    outdir = makedir(args.outdir)

    emph(f"{n}+1 qubits")

    H = TFIM(n)
    psi_target = groundstate(H, n)
    rho_target = psi_target[:, None] * psi_target[None, :].conj()
    rho_target_0 = rho_target

    add_random_ancilla = AddRandomAncilla(0)
    restrict = RestrictMeasurementOutcome(0)

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    print(vars(args))
    torch.save(H, f"{outdir}/H.pt")
    torch.save(psi_target, f"{outdir}/psi_target.pt")

    targets = []
    blocks = deque([])

    ublock = UnitaryBlock(H, l=l, use_trotter=False, n=n).set_direction_forward()

    if args.reload != "":
        prev_args = torch.load(args.compare_args)
        for key in vars(prev_args):
            if key == "reload" or key == "compare_args":
                continue
            else:
                assert vars(args)[key] == vars(prev_args)[key]

        print(vars(args))
        print(vars(prev_args))

        start_epoch = int(args.reload.split("-")[-1].split(".")[0])
        rho_target = torch.load(args.reload)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        optimizer = optim.Adam(ublock.parameters(), lr=0.01)

        # torch.save(optimizer.state_dict(), f"{outdir}/optimizer_{epoch}.pt")
        torch.save(rho_target, f"{outdir}/rho_t-{epoch}.pt")

        targets.append(rho_target)
        rho_out_ = add_random_ancilla.apply_to_density_matrix(rho_target)

        block = Block(unitaryblock=ublock)

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            rho_in_ = ublock.do_backward(ublock.apply_to_density_matrix, rho_out_)
            rho_in = restrict.apply_to_density_matrix(rho_in_)
            value1 = rho_in.trace().real
            value2 = HS(block.apply_to_density_matrix(rho_target_0), rho_target_0).real

            value = value1 + value2

            loss = -value
            loss.backward()

            optimizer.step()

            value1, value2 = retrieve(value1, value2)
            print(f"{value1:.6f} {value2:.6f}")

            with open(f"{outdir}/values.txt", "a") as f:
                f.write(f"{epoch} {i} Ancilla {value1} invariance {value2}\n")

        rho_target = (rho_in / rho_in.trace().real).detach()

        torch.save(block, f"{outdir}/block_t-{epoch}.pt")
        torch.save(block.state_dict(), f"{outdir}/params_t-{epoch}.pt")

        #################################################################################

        # blocks.appendleft(block)

        # circuit = Channel(gates=blocks)
        # torch.save(circuit, f"{outdir}/circuit_{epoch+1}.pt")
        # torch.save(circuit.state_dict(), f"{outdir}/circuit_params_{epoch}.pt")

        # circuitvalue = testcircuit(circuit, psi_target)
        # emph(f"After {epoch+1} epochs: circuit value {circuitvalue:.5f}")

        # with open(f"{outdir}/circuitvalues.txt", "a") as f:
        #    f.write(f"{epoch} {circuitvalue}\n")

        # ublock = copy.deepcopy(ublock)
