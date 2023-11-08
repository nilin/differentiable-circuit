import argparse
import cudaswitch

from jax.config import config

config.update("jax_enable_x64", True)

cuda_on = cudaswitch.cuda_on
print(f"CUDA {'_ON_' if cuda_on else '_OFF_'}")
