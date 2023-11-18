import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_gate_implementation():
    import gate_implementation

    return gate_implementation.TorchGate()


gen = torch.Generator(device=device)
