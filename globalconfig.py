from cudaswitch import numba_CUDA_on
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
args, _ = argparser.parse_known_args()


"""pick gate implementation"""
if args.torch and args.numba:
    raise ValueError("Cannot request both torch and numba")

implementation = "numba" if args.numba else "torch"


"""CUDA for torch"""
torchdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch using device:", torchdevice)
