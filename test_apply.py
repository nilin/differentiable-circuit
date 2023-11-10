from config import device
import config
import torch
import argparse
from differentiable_circuit import Params, overlap
from typing import Literal
import examples


def get_state(L, seed=0):
    N = 2**L
    x = torch.normal(
        0, 1, (2, N), generator=torch.Generator().manual_seed(seed)
    )
    x = torch.complex(x[0], x[1]).to(config.tcomplex)
    x = x.to(device)
    x = x / torch.norm(x)
    return x

def test_apply(L):
    print("test apply")

    tau, zeta = Params.def_param(0.1, 0.2)
    C, *_ = examples.Block(L, 2, 1.0, tau, zeta)
    x = examples.zero_state(L)

    y = C.apply(x)
    z = C.apply(y, reverse=True)

    print("norm of output", torch.norm(y).detach().cpu().numpy())
    print("dist", torch.norm(z - x).detach().cpu().numpy())


def test_grad(L, depth):
    def getcircuitwithparams(tau, zeta):
        C = examples.Block(L, depth, 1.0, tau, zeta)
        x = examples.zero_state(L)
        target = get_state(L, seed=0)

        def Obs(y):
            return target * overlap(target, y)

        def getloss(x):
            y = C.apply(x)
            return torch.abs(overlap(target, y)) ** 2

        return C, x, Obs, getloss

    def get_grad(tau, zeta, mode: Literal["qcontrol", "slow_param_shift"]):
        match mode:
            case "qcontrol":
                C, x, Obs, getloss = getcircuitwithparams(tau, zeta)
                C.optimal_control(x, Obs=Obs)
                return tau.grad, zeta.grad

            case "slow_param_shift":
                eps = 0.0001
                C, x, Obs, getloss1 = getcircuitwithparams(tau, zeta)
                C, x, Obs, getloss2 = getcircuitwithparams(tau + eps, zeta)
                C, x, Obs, getloss3 = getcircuitwithparams(tau, zeta + eps)

                def estimate(getloss1, getloss2):
                    return (
                        ((getloss2(x) - getloss1(x)).real / eps).detach().cpu()
                    )

                return estimate(getloss1, getloss2), estimate(
                    getloss1, getloss3
                )

    tau, zeta = Params.def_param(0.1, 0.2)
    print(get_grad(tau, zeta, mode="qcontrol"))
    print(get_grad(tau, zeta, mode="slow_trace"))
    print(get_grad(tau, zeta, mode="slow_param_shift"))

    return tau, zeta


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--L", type=int, default=2)
    argparser.add_argument("--depth", type=int, default=1)
    args, _ = argparser.parse_known_args()

    test_grad(args.L, args.depth)
