#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import numpy as np
from datatypes import torchcomplex, show
from gate_implementation_anysize import apply_gate, apply_sparse_gate, device
import time


# In[11]:


def test_correctness():
    psi0 = torch.arange(8, device=device) + 1
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    XZ = np.kron(X, Z)
    XZ = torchcomplex(XZ).to(device)
    psi1 = apply_gate([1, 2], XZ, psi0)
    np.testing.assert_allclose(
        psi1.cpu(), torch.tensor([3, -4, 1, -2, 7, -8, 5, -6]).cpu(), atol=0.00001
    )
    print("Correctness test passed for dense")

    XZ_sparse = (
        [(0, 2, 1.0), (1, 3, -1.0), (2, 0, 1.0), (3, 1, -1.0)],
        torchcomplex(np.array([1.0, -1.0, 1.0, -1.0])),
    )
    psi2 = apply_sparse_gate([1, 2], XZ_sparse, psi0)
    np.testing.assert_allclose(
        psi2.cpu(), torch.tensor([3, -4, 1, -2, 7, -8, 5, -6]).cpu(), atol=0.00001
    )
    print("Correctness test passed for sparse")


# In[12]:


class PerfTest:
    @staticmethod
    def run(n, positions, sparse=False):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        if isinstance(positions, int):
            positions = list(range(positions))
        k = len(positions)
        if sparse:
            matrix = PerfTest.get_k_qubit_sparse_Xk(k)
        else:
            matrix = PerfTest.get_k_qubit_dense_Hadamard(k)
        psi0 = PerfTest.get_state(n)
        if device.type == "cuda":
            start.record()
            PerfTest.apply_gate_to_qubits(positions, matrix, psi0, sparse=sparse)
            end.record()
            """https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/10"""
            torch.cuda.synchronize()
        else:
            t0 = time.time()
            PerfTest.apply_gate_to_qubits(positions, matrix, psi0, sparse=sparse)
            t1 = time.time()
        if device.type == "cuda":
            s = start.elapsed_time(end) / 1000
        else:
            s = t1 - t0
        if sparse:
            print(f"{s:.4f} s for {k} qubit gate on {n} qubit state using sparsity")
        else:
            print(f"{s:.4f} s for general {k} qubit gate on {n} qubit state")

    @staticmethod
    def apply_gate_to_qubits(positions, matrix, psi0, sparse=False):
        if sparse:
            return apply_sparse_gate(positions, matrix, psi0)
        else:
            return apply_gate(positions, matrix, psi0)

    @staticmethod
    def get_state(n):
        return torch.arange(2**n, device=device)

    @staticmethod
    def get_k_qubit_dense_Hadamard(k):
        X = torchcomplex(np.array([[1, 1], [1, -1]]) / np.sqrt(2)).to(device)
        X_ = torchcomplex(np.array([[1]])).to(device)
        for _ in range(k):
            X_ = torch.kron(X_, X)
        return X_

    @staticmethod
    def get_k_qubit_sparse_Xk(k):
        K = 2**k
        I = torch.arange(K, device=device)
        J = K - I
        V = torch.ones(K, device=device)
        Xk = torch.stack([I, J, V], dim=1)
        return Xk


# In[13]:


test_correctness()


# In[ ]:
