from differentiable_circuit import *


class Channel(Circuit):
    def apply(self, psi: State, randomness: Iterable[uniform01] = [], register=False):
        outcomes = []
        p_conditional = []
        checkpoints = []
        randomness = deque(randomness)
        for gate, where in self.flatgates_and_where():
            if not isinstance(gate, Measurement):
                psi = gate.apply(psi)
            else:
                if register:
                    checkpoints.append(psi)

                u = randomness.popleft()
                psi, m, p = gate.apply(psi, u=u, normalize=True)
                outcomes.append(m)
                p_conditional.append(p.cpu())

        if register:
            return psi, outcomes, p_conditional, checkpoints
        else:
            return psi

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[State, State] = None,
        outcome_values: List[float] = None,
        randomness: Iterable[uniform01] = [],
    ):
        psi_t, o, p, ch = self.apply(psi, randomness, register=True)

        E = 0

        if Obs is None:
            Xt = torch.zeros_like(psi_t)
        else:
            E += Obs(psi_t)
            (Xt,) = torch.autograd.grad(E, psi_t, retain_graph=True)
            Xt = Xt.conj()

        if outcome_values is None:
            dVal_dp = [0.0] * len(o)
        else:
            E += sum([o_i * v_i for o_i, v_i in zip(o, outcome_values)])
            dVal_dp = outcome_values

        return E, p, self.backprop(psi_t, Xt, dVal_dp, o, p, ch)

    def backprop(self, psi, X, dObs_dp, outcomes, p_conditional, checkpoints):
        dE_inputs_rev = []
        inputs_rev = []

        for gate, where in self.flatgates_and_where()[::-1]:
            if isinstance(gate, ThetaGate):
                psi = gate.apply_reverse(psi)

                dU = gate.dgate_state()
                dE_input = cdot(X, gate.apply_gate_state(dU, psi)).real
                X = gate.apply_reverse(X)

                dE_inputs_rev.append(dE_input)
                inputs_rev.append(gate.input)

            elif isinstance(gate, Measurement):
                psi = checkpoints.pop()
                m = outcomes.pop()
                p = p_conditional.pop()
                X = gate.apply_reverse(X, m) / torch.sqrt(p)

                dvdp = dObs_dp.pop()
                X += dvdp * 2 * psi.conj()

            else:
                psi = gate.apply_reverse(psi)
                X = gate.apply_reverse(X)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X
