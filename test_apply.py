from config import device
import torch
import argparse
from gates import UX, UZZ
from differentiable_circuit import UnitaryCircuit, Params, overlap
from typing import Literal


def get_problem(L, depth, theta, phi):
    print(f"{L} qubits, vector size {2**L:,}")

    Xs = [UX(theta / depth, i) for i in range(L)]
    ZZs = [UZZ(phi / depth, i, i + 1) for i in range(L - 1)]
    layer = Xs + ZZs
    C = [op for k in range(depth) for op in layer]

    return UnitaryCircuit(C)


def get_state(L, seed=0):
    N = 2**L
    x = torch.normal(
        0, 1, (2, N), generator=torch.Generator().manual_seed(seed)
    )
    x = torch.complex(x[0], x[1]).to(torch.complex64)
    x = x.to(device)
    x = x / torch.norm(x)
    return x


def test_apply(L):
    theta, phi = Params.def_param(0.1, 0.2)
    C, *_ = get_problem(L, 2, theta, phi)
    x = get_state(L)

    y = C.apply(x)
    z = C.apply(y, reverse=True)

    print("norm of output", torch.norm(y).detach().cpu().numpy())
    print("dist", torch.norm(z - x).detach().cpu().numpy())


def test_grad(L, depth):
    def getcircuitwithparams(theta, phi):
        C = get_problem(L, depth, theta, phi)
        x = get_state(L, seed=0)
        target = get_state(L, seed=1)

        def Obs(y):
            return target * overlap(target, y)

        def getloss(x):
            y = C.apply(x)
            return torch.abs(overlap(target, y)) ** 2

        return C, x, Obs, getloss

    def get_grad(theta, phi, mode: Literal["qcontrol", "slow_param_shift"]):
        match mode:
            case "qcontrol":
                C, x, Obs, getloss = getcircuitwithparams(theta, phi)
                C.optimal_control(x, Obs=Obs)
                return theta.grad, phi.grad

            case "slow_param_shift":
                eps = 0.0001
                C, x, Obs, getloss1 = getcircuitwithparams(theta, phi)
                C, x, Obs, getloss2 = getcircuitwithparams(theta + eps, phi)
                C, x, Obs, getloss3 = getcircuitwithparams(theta, phi + eps)

                def estimate(getloss1, getloss2):
                    return (
                        ((getloss2(x) - getloss1(x)).real / eps).detach().cpu()
                    )

                return estimate(getloss1, getloss2), estimate(
                    getloss1, getloss3
                )

    theta, phi = Params.def_param(0.1, 0.2)
    print(get_grad(theta, phi, mode="qcontrol"))
    print(get_grad(theta, phi, mode="slow_trace"))
    print(get_grad(theta, phi, mode="slow_param_shift"))

    return theta, phi


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=2)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    test_grad(args.L, args.depth)
