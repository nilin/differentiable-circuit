from config import device
import torch
import argparse
from gates import UX, UZZ
from differentiable_circuit import UnitaryCircuit


def get_circuit(L, nlayers):
    print(f"{L} qubits, vector size {2**L:,}")

    nlayers = 1

    theta = torch.tensor(0.1).requires_grad_()
    phi = torch.tensor(0.1).requires_grad_()

    Xs = [UX(theta + phi, i) for i in range(L)]
    ZZs = [UZZ(theta, i, i + 1) for i in range(L - 1)]
    layer = Xs + ZZs
    C = [op for k in range(nlayers) for op in layer]

    return UnitaryCircuit(C), dict(theta=theta, phi=phi)


def get_state(L):
    N = 2**L
    r = torch.normal(torch.zeros(N), torch.ones(N))
    i = torch.normal(torch.zeros(N), torch.ones(N))
    x = torch.complex(r, i).to(torch.complex64)
    x = x.to(device)
    x = x / torch.norm(x)
    return x


def test_apply(L):
    C, *_ = get_circuit(L, 2)
    x = get_state(L)

    y = C.apply(x)
    z = C.apply(y, reverse=True)

    print("norm of output", torch.norm(y).detach().cpu().numpy())
    print("dist", torch.norm(z - x).detach().cpu().numpy())


def test_grad(L):
    C, params = get_circuit(L, 2)
    x = get_state(L)
    target = get_state(L)

    def Op(y):
        return target.conj().dot(y) * target

    C.loss_and_grad(x, Op=Op)

    theta = params["theta"]
    phi = params["phi"]

    print(theta.grad.detach().cpu().numpy())
    print(phi.grad.detach().cpu().numpy())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=20)
    args, _ = argparser.parse_known_args()

    test_grad(args.L)
