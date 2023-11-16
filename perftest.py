import torch
import config
import numpy as np
from datatypes import torchcomplex, show
from gate_implementation import AnySizeGate
from dataclasses import dataclass
from typing import List
import time


def gatematrix(gate_as_lists):
    return torchcomplex(gate_as_lists)


def test_correctness():
    n = 3
    psi0 = torch.arange(8, device=config.device) + 1

    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    XZ = np.kron(X, Z)
    XZ = torchcomplex(XZ).to(config.device)

    psi1 = AnySizeGate.apply_gate_to_qubits([1, 2], XZ, psi0)
    np.testing.assert_allclose(
        psi1.cpu(), torch.tensor([3, -4, 1, -2, 7, -8, 5, -6]).cpu(), atol=0.00001
    )


class PerfTest:
    @staticmethod
    def run(n, k, diag=False):
        if config.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        kwargs = PerfTest.prep_test(n, k, diag)

        if config.device.type == "cuda":
            start.record()
            AnySizeGate.apply_gate_to_qubits(**kwargs)
            end.record()

            """https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/10"""
            torch.cuda.synchronize()

        else:
            t0 = time.time()
            AnySizeGate.apply_gate_to_qubits(**kwargs)
            t1 = time.time()

        if config.device.type == "cuda":
            s = start.elapsed_time(end) / 1000
        else:
            s = t1 - t0

        if diag:
            print(f"{s:.4f} s for diagonal {k} qubit gate on {n} qubit state")
        else:
            print(f"{s:.4f} s for general {k} qubit gate on {n} qubit state")

    @staticmethod
    def prep_test(n, k, diag):
        psi0 = torch.arange(2**n, device=config.device)
        if not diag:
            X = torchcomplex(np.array([[0, 1], [1, 0]])).to(config.device)
            X_ = torchcomplex(np.array([[1]])).to(config.device)
            for _ in range(k):
                X_ = torch.kron(X_, X)

            return dict(
                positions=list(range(k)), k_qubit_matrix=X_, psi=psi0, diag=False
            )

        if diag:
            D = torchcomplex(torch.arange(2**k, device=config.device))
            return dict(list(range(k)), D, psi0, diag=True)


if __name__ == "__main__":
    test_correctness()
    ms = PerfTest.run(20, 5)
    ms = PerfTest.run(30, 5)
