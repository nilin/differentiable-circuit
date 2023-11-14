import argparse
from test import TestGradChannel
from examples import *
from torch import optim
import os
from torch.nn import Parameter, ParameterList


class Problem(TestGradChannel):
    def __init__(self, n=6):
        self.n = n

        # zetas1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        # taus1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        # self.params = ParameterList(zetas1 + taus1)
        # self.circuit = Block(self.n, taus1, zetas1)

        zetas1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus1 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus2 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        zetas3 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        taus3 = [Parameter(torch.tensor(1.0)), Parameter(torch.tensor(1.0))]
        self.params = ParameterList(taus1 + zetas1 + taus2 + zetas2 + taus3 + zetas3)

        self.H = TFIM(self.n)
        B1 = Block(self.H, taus1, zetas1)
        B2 = Block(self.H, taus2, zetas2)
        B3 = Block(self.H, taus3, zetas3)
        self.circuit = CircuitChannel(B1.gates + B2.gates + B3.gates)

        self.prepstates()

    def prepstates(self):
        self.psi0 = zero_state(self.n + 1)
        self.target = self.groundstate()
        self.Obs = lambda y: self.target * cdot(self.target, y)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n", type=int, default=2)
    argparser.add_argument("--mode", type=str, default="oc")
    args, _ = argparser.parse_known_args()
    n = args.n
    mode = args.mode

    def bar():
        print(80 * "_" + "\n")

    bar()
    print(f"{n}+1 qubits")
    bar()

    np.random.seed(0)
    torch.manual_seed(1)

    print("preparing problems")
    problem = Problem(n)

    optimizer = optim.Adam(problem.params, lr=0.01)

    iterations = 100
    randomness = torch.rand((iterations, 3))
    values = []

    os.makedirs("_outputs", exist_ok=True)

    for i, rand in enumerate(randomness):
        optimizer.zero_grad()

        match mode:
            case "oc":
                value, _ = problem.circuit.optimal_control(
                    problem.psi0, problem.Obs, randomness=rand
                )
            case "dm":
                rho = problem.psi0[:, None] * problem.psi0[None, :].conj()
                rho_out = problem.circuit.apply_to_density_matrix(rho)
                value = cdot(problem.target, rho_out @ problem.target).real

        loss = -value
        loss.backward()
        optimizer.step()

        print(value)
        values.append(value.detach().cpu().numpy())

        with open(f"_outputs/{mode}_values.txt", "a") as f:
            f.write(f"{value.detach().cpu().numpy()}\n")

        if i % 100 == 0:
            import matplotlib.pyplot as plt

            plt.plot(values)
            plt.ylim(-1, 1)
            plt.savefig(f"_outputs/{mode}_values.png")
