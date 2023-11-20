import argparse
from examples import *
from non_unitary_gates import *
from torch import optim
import os
from differentiable_channel import *
from datatypes import *
import torch
import json
import gate_implementation
from gate_implementation import add_qubits


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
    rho_t, checkpoints = circuit.apply_to_density_matrix(
        I, checkpoint_at=lambda gate: isinstance(gate, Measurement)
    )
    value = cdot(target, rho_t @ target).real
    # print(f"Circuit value: {value:.5f}")
    return value, checkpoints


def retrieve(*values):
    for value in values:
        yield value.detach().cpu().numpy()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dm", action="store_true")
    argparser.add_argument("--n", type=int, default=6)
    argparser.add_argument("--l", type=int, default=6)
    argparser.add_argument("--trottersteps", type=int, default=1)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--iterations", type=int, default=10000)
    argparser.add_argument("--iterations_per_epoch", type=int, default=100)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    outdir = makedir(args.outdir)

    emph(f"{n}+1 qubits")

    H = TFIM(n)
    psi_target = groundstate(H, n)

    add_ancilla = AddAncilla(0)
    add_random_ancilla = AddRandomAncilla(0)
    measure = Measurement(0)

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(H, f"{outdir}/H.pt")

    targets = []
    blocks = deque([])

    ublock = UnitaryBlock(H, l=l, use_trotter=False, n=n).set_direction_forward()

    optimizer = optim.Adam(ublock.parameters(), lr=0.01)

    block = Block(unitaryblock=ublock)

    def Obs(psi_out):
        psi_0, psi_1 = measure.apply_both(psi_out, normalize=False)
        return squared_overlap(psi_target, psi_0) + squared_overlap(psi_target, psi_1)

    for i in range(args.iterations):
        optimizer.zero_grad()

        psi_in = add_ancilla.apply(psi_target)
        # value1, _ = ublock.optimal_control(psi_in, Obs)
        value1 = Obs(ublock.apply(psi_in))

        psi_out = psi_target
        psi_out_0 = add_qubits((0,), (1.0, 0.0), psi_out)
        psi_out_1 = add_qubits((0,), (0.0, 1.0), psi_out)

        psi_in_0 = ublock.do_backward(ublock.apply, psi_out_0)
        psi_in_1 = ublock.do_backward(ublock.apply, psi_out_1)

        value2 = (measure.probability(psi_in_0, 0) + measure.probability(psi_in_1, 0)) / 2

        value = value1 + value2
        loss = -value
        loss.backward()

        optimizer.step()

        value1, value2 = retrieve(value1, value2)
        print(f"{value1:.6f} {value2:.6f}")

        with open(f"{outdir}/values.txt", "a") as f:
            f.write(f"{i} Ancilla {value1} invariance {value2}\n")

        # rho_target = (rho_in_restricted / rho_in_restricted.trace().real).detach()

        # torch.save(ublock, f"{outdir}/block_t-{epoch}.pt")
        # torch.save(ublock.state_dict(), f"{outdir}/params_t-{epoch}.pt")

        if i % args.iterations_per_epoch == 0:
            psi_test = HaarState(n, config.gen).pure_state()
            testvals = []

            for d in range(10):
                psi_test = block.apply(psi_test, torch.rand(1))
                testvals.append(squared_overlap(psi_target, psi_test))
                testvalstring = "\n".join([f"{v:.5f}" for v in testvals])

            emph(f"Test values: {testvalstring}")

            # circuit = Channel(gates=blocks)
            # torch.save(circuit, f"{outdir}/circuit_{epoch+1}.pt")
            # torch.save(circuit.state_dict(), f"{outdir}/circuit_params_{epoch}.pt")

            # circuitvalue, checkpoints = testcircuit(circuit, psi_target)
            # emph(f"After {epoch+1} epochs: circuit value {circuitvalue:.5f}")

            # ublock = copy.deepcopy(ublock)
