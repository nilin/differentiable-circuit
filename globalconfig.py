import argparse
from numba import cuda
from jax.config import config
import torch

config.update("jax_enable_x64", True)
config.update("jax_platforms", "cpu")


argparser = argparse.ArgumentParser()
argparser.add_argument("--torch", action="store_true")
argparser.add_argument("--numba", action="store_true")
argparser.add_argument("--nocuda", action="store_true")
args = argparser.parse_args()


"""pick gate implementation"""
if args.torch and args.numba:
    raise ValueError("Cannot request both torch and numba")

implementation = "torch" if args.torch else "numba"


"""CUDA for numba"""
numba_CUDA_on = (not args.nocuda) and cuda.is_available()
print(f"CUDA {'ON' if numba_CUDA_on else 'OFF'} for numba")

"""CUDA for torch"""
torchdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch using device:", torchdevice)
