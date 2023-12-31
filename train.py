import argparse
from torch import optim
import os
import math
import torch
import json
from differentiable_circuit.examples import *
from differentiable_circuit.non_unitary_gates import *
from differentiable_circuit.datatypes import *
from differentiable_circuit import gate_implementation
from differentiable_circuit.gate_implementation import add_qubits
from differentiable_circuit.circuit import (
    Circuit,
    UnitaryCircuit,
    Non_unitary_circuit,
    SquaredOverlap,
)


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


def testcircuit(circuit, target, checkpoints=[1, 5, 10, 25, 50, 75, 100]):
    haarstate = HaarState(n, config.gen)
    psi_in = haarstate.pure_state().detach()
    for i in range(checkpoints[-1]):
        psi_out = circuit.apply(psi_in)
        value = squared_overlap(target, psi_out)
        if i + 1 in checkpoints:
            yield i + 1, value.detach().cpu().numpy()


def evaluation(circuit, psi_target):
    print("\nevaluation")
    for s in range(3):
        print(f"\nsample {s+1}")
        for i, val in testcircuit(circuit, psi_target):
            print(f"{i} repeats {val:.6f}")
    print("evaluation done\n")


def retrieve(*values):
    for value in values:
        yield value.detach().cpu().numpy()


class Rewinder(Circuit):
    def apply(self,psi):
        raise ValueError("Rewinder does not support forward apply")

    def rewind(self, psi, betas):
        N = len(psi)
        for gate in self.gates[::-1]:
            psi_ = gate_implementation.add_qubits((0,), betas.pop(0), psi)
            psi_ = gate.do_backward(gate.apply, psi_)
            psi = psi[:N]
        if betas:
            return gate_implementation.add_qubits((0,), betas.pop(0), psi)
        else:
            return psi


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--customgrad", action="store_true")
    argparser.add_argument("--n", type=int, default=10)
    argparser.add_argument("--l", type=int, default=20)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--iterations_per_epoch", type=int, default=250)
    argparser.add_argument("--outdir", type=str, default="_outputs/run")
    args, _ = argparser.parse_known_args()

    n = args.n
    l = args.l
    outdir = makedir(args.outdir)

    emph(f"{n}+1 qubits")

    H = TFIM(n)
    psi_target = groundstate(H, n)

    add_ancilla = Add_0_ancilla(0)
    measure = Measurement(0)

    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=2)
    torch.save(H, f"{outdir}/H.pt")

    H_shifted = H.to_dense(n).set_ignored_positions((0,))

    circuit = Non_unitary_circuit()
    rewinder = Rewinder()

    def Obs1_and_grad(psi):
        E = probabilitymass(psi[: 2**n])
        grad0 = psi.conj()[: 2**n]
        grad = torch.cat((grad0, torch.zeros_like(grad0)))
        return E, -grad

    def Obs2_and_grad(psi):
        psi0 = psi[: 2**n]
        psi1 = psi[2**n :]
        value = squared_overlap(psi_target, psi0) + squared_overlap(psi_target, psi1)
        grad = torch.cat(
            (
                cdot(psi0, psi_target) * psi_target.conj(),
                cdot(psi1, psi_target) * psi_target.conj(),
            )
        )
        return value, -grad

    Obs1 = lambda psi: Obs1_and_grad(psi)[0]
    Obs2 = lambda psi: Obs2_and_grad(psi)[0]

    for epoch in range(args.epochs):
        torch.save(circuit, f"{outdir}/circuit-{epoch}.pt")
        ublock = UnitaryBlock(H_shifted=H_shifted, l=l).set_direction_forward()
        optimizer = optim.Adam(ublock.parameters(), lr=0.01)

        for i in range(args.iterations_per_epoch):
            optimizer.zero_grad()

            betas = [[(1, 0), (0, 1)][np.random.choice(2)] for _ in range(epoch + 1)]
            psi_target_ = rewinder.rewind(psi_target, betas)

            if args.customgrad:
                psi_0, value1, _ = ublock.do_backward(
                    ublock.optimal_control, psi_target_, Obs1_and_grad
                )
            else:
                psi_0 = ublock.do_backward(ublock.apply, psi_target_)
                value1 = Obs1(psi_0)

            y = add_qubits((0,), (1, 0), psi_target)

            if args.customgrad:
                _, value2, _ = ublock.optimal_control(y, Obs2_and_grad)
            else:
                y = ublock.apply(y)
                value2 = Obs2(y)

            value = value1 + value2

            if not args.customgrad:
                loss = -value
                loss.backward()

            optimizer.step()

            value1, value2 = retrieve(value1, value2)
            print(f"{i} {value1:.6f} {value2:.6f}")

            with open(f"{outdir}/values.txt", "a") as f:
                f.write(f"{i} Ancilla {value1} invariance {value2}\n")

        torch.save(ublock.state_dict(), f"{outdir}/params_t-{epoch}.pt")
        circuit.gates.insert(0, Block(ublock))
        evaluation(circuit, psi_target)

        rewinder.gates.insert(0,ublock)
