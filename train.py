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


def testcircuit(circuit, target, nsamples=10, repeats=20):
    haarstate = HaarState(n, config.gen)
    values = []
    print(f"evaluation")
    for i in range(nsamples):
        psi_in = haarstate.pure_state().detach()
        for _ in range(repeats):
            psi_out = circuit.apply(psi_in)
        value = squared_overlap(target, psi_out)
        values.append(value.detach().cpu().numpy())
    print(f"evaluation value {np.mean(np.array(values))}")


def retrieve(*values):
    for value in values:
        yield value.detach().cpu().numpy()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dm", action="store_true")
    argparser.add_argument("--n", type=int, default=6)
    argparser.add_argument("--l", type=int, default=20)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--iterations_per_epoch", type=int, default=1000)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    outdir = makedir(args.outdir)

    emph(f"{n}+1 qubits")

    H = TFIM(n)
    psi_target = groundstate(H, n)

    add_ancilla = Add_0_ancilla(0)
    random_out_ancilla = Random_out_ancilla(0)
    measure = Measurement(0)

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(H, f"{outdir}/H.pt")

    targets = []

    H_shifted = H.to_dense(n).set_ignored_positions((0,))

    # def Obs(psi_out):
    #    psi_0, psi_1 = measure.apply_both(psi_out, normalize=False)
    #    return squared_overlap(psi_target, psi_0) + squared_overlap(psi_target, psi_1)

    circuit = Non_unitary_circuit(gates=[])
    backcircuit = Non_unitary_circuit(gates=[])

    for epoch in range(args.epochs):
        torch.save(circuit, f"{outdir}/circuit-{epoch}.pt")
        ublock = UnitaryBlock(H_shifted=H_shifted, l=l).set_direction_forward()
        block = Block(ublock)
        backblock = Random_out_ancilla_block(ublock)
        optimizer = optim.Adam(ublock.parameters(), lr=0.01)

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            with torch.no_grad():
                beta = HaarState(2, config.gen).pure_state()
                psi_target_inner = backcircuit.do_backward(backcircuit.apply, psi_target)
                psi_target_inner = random_out_ancilla.apply_backward(psi_target_inner)

            psi_in = ublock.do_backward(ublock.apply, psi_target_inner)
            value1 = probabilitymass(psi_in[: 2**n])

            y = add_ancilla.apply(psi_target)
            y = ublock.apply(y)
            psi0, psi1, p0, p1 = measure.both_outcomes(y)

            value2 = p0 * squared_overlap(psi_target, psi0) + p1 * squared_overlap(
                psi_target, psi1
            )
            # value1 = value2

            value = value1 + 10 * value2
            loss = -value
            loss.backward()

            optimizer.step()

            value1, value2 = retrieve(value1, value2)
            print(f"{i} {value1:.6f} {value2:.6f}")

            with open(f"{outdir}/values.txt", "a") as f:
                f.write(f"{i} Ancilla {value1} invariance {value2}\n")

        torch.save(ublock.state_dict(), f"{outdir}/params_t-{epoch}.pt")
        circuit.gates.insert(0, block)
        backcircuit.gates.insert(0, backblock)

        testcircuit(circuit, psi_target)
