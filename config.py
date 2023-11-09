import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tcomplex = torch.complex64


def get_default_gate_implementation():
    import gate_implementations

    return gate_implementations.TorchGate()
