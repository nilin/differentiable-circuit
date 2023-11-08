"""
easily switch between cuda and non-cuda
the purpose of this snippet is to construct the decorator:

@def new_value_at_i(signature)

which takes a device function and parallelizes it with cuda if it is available and otherwise with prange
"""
from globalconfig import numba_CUDA_on
import numpy as np
from numba import cuda, prange, void, float64, njit, uint64
import numba


blocks = 1000
threads = 1000


def cudaswitch_kernel(dtypes):
    def dec(f):
        F = cuda.jit(dtypes)(f) if numba_CUDA_on else njit(dtypes, parallel=True)(f)
        return F[blocks, threads] if numba_CUDA_on else F

    return dec


def cudaswitch_device(dtypes):
    def dec(f):
        return cuda.jit(dtypes, device=True)(f) if numba_CUDA_on else njit(dtypes)(f)

    return dec


def makeboth_device(f, dtypes):
    fname = f.__name__
    globals()[f.__name__] = cudaswitch_device(dtypes)(f)
    globals()[f.__name__ + "_nevercuda"] = njit(dtypes)(f)


def get_optionalcuda_parallelized(signature, body):
    argnum = len(signature._args)

    jittedbody = cudaswitch_device(signature)(body)

    argstring = ",".join(["x{}".format(i) for i in range(argnum - 1)])
    code = "\
def F({0},D):                               \n\
    if numba_CUDA_on:                       \n\
        start = cuda.grid(1)                \n\
        stepsize = cuda.gridsize(1)         \n\
        for i in range(start, D, stepsize): \n\
            body({0},i)                     \n\
    else:                                   \n\
        for i in prange(D):                 \n\
            body({0},i)                     \n\
    ".format(
        argstring
    )
    with open("_cudaswitch_code.py", "w") as f:
        f.write(code)

    tempdict = {"body": jittedbody}
    globals()["body"] = jittedbody
    exec(code, globals(), tempdict)
    return cudaswitch_kernel(signature)(tempdict["F"])


####################################################################################################

"""
When we use the decorator

@new_value_at_i(void(float64[:], uint64))
def square(X, i):
    X[i] = X[i] ** 2

The last entry in the definition corresponds to index.
The resulting function is of the form square(X, D), 
where D is the length of X.

"""


def new_value_at_i(signature):
    def dec(body):
        return get_optionalcuda_parallelized(signature, body)

    return dec


####################################################################################################


def try_device_array_like(x):
    if numba_CUDA_on:
        return cuda.device_array_like(x)
    else:
        return np.zeros_like(x)


def try_device_array(length, dtype=numba.complex64):
    if numba_CUDA_on:
        return cuda.device_array(length, dtype=dtype)
    else:
        return np.zeros(length, dtype=dtype)


def try_to_device(x):
    if numba_CUDA_on:
        return cuda.to_device(x)
    else:
        return x


def wrapcuda(f):
    def f_(*args):
        moved = [isinstance(x, np.ndarray) and numba_CUDA_on for x in args]
        deviceargs = [cuda.to_device(x) if m else x for m, x in zip(moved, args)]
        f(*deviceargs)
        args = [x.copy_to_host() if m else x for m, x in zip(moved, deviceargs)]
        return args

    return f_


if __name__ == "__main__":

    @new_value_at_i(void(float64[:], uint64))
    def f(X, i):
        X[i] = X[i] ** 2

    Y = wrapcuda(f)(np.arange(10**6).astype(float), 10**6)
    print(Y)
