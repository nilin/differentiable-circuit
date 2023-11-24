from differentiable_circuit import *


class Non_unitary_circuit(Circuit):
    def apply_and_register(self, psi: State):
        outcomes = []
        p_conditional = []
        checkpoints = []
        randomness = deque(self.make_randomness())

        for gate, where in self.flatgates_and_where():
            if not isinstance(gate, Measurement):
                psi = gate.apply(psi)
            else:
                checkpoints.append(psi)
                u = randomness.popleft()
                psi, m, p = gate.measure(psi, u=u)
                outcomes.append(m)
                p_conditional.append(p.cpu())

        return psi, outcomes, p_conditional, checkpoints

    def optimal_control(
        self,
        psi: State,
        Obs: Callable[State, State] = None,
    ):
        psi_t, o, p, ch = self.apply_and_register(psi)

        E = 0

        if Obs is None:
            Xt = torch.zeros_like(psi_t)
        else:
            E += Obs(psi_t)
            (Xt,) = torch.autograd.grad(E, psi_t, retain_graph=True)
            Xt = Xt.conj()

        return E, p, self.backprop(psi_t, Xt, o, p, ch)

    def backprop(self, psi, X, outcomes, p_conditional, checkpoints):
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
                X = gate.unmeasure(X, m) / torch.sqrt(p)

            else:
                psi = gate.apply_reverse(psi)
                X = gate.apply_reverse(X)

        torch.autograd.backward(inputs_rev, dE_inputs_rev)
        return X

    def make_randomness(self):
        nmeasurements = len(
            [
                gate
                for gate, where in self.flatgates_and_where()
                if isinstance(gate, Measurement)
            ]
        )
        return torch.rand(nmeasurements)
