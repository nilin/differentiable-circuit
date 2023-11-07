# easily switch between cuda and non-cuda
# the purpose of this snippet is to construct the decorator:
#
# @def new_value_at_i(signature)
#
# which takes a device function and parallelizes it with cuda if it is available and otherwise with prange


import numpy as np
from numba import cuda, prange, void, float64, njit, uint64
import sys


cuda_on='nocuda' not in sys.argv and cuda.is_available()
print('CUDA {}'.format(['OFF','ON'][cuda_on]))

blocks=1000
threads=1000

def cudaswitch_kernel(dtypes):
    def dec(f):
        F = cuda.jit(dtypes)(f) if cuda_on else njit(dtypes,parallel=True)(f)
        return F[blocks,threads] if cuda_on else F
    return dec

def cudaswitch_device(dtypes):
    def dec(f):
        return cuda.jit(dtypes,device=True)(f) if cuda_on else njit(dtypes)(f)
    return dec


def makeboth_device(f,dtypes):
    fname=f.__name__
    globals()[f.__name__]=cudaswitch_device(dtypes)(f)
    globals()[f.__name__+'_nevercuda']=njit(dtypes)(f)


def get_optionalcuda_parallelized(signature,body):

    argnum=len(signature._args)

    jittedbody=cudaswitch_device(signature)(body)

    argstring=','.join(['x{}'.format(i) for i in range(argnum-1)])
    code='\
def F({0},D):                               \n\
    if cuda_on:                             \n\
        start = cuda.grid(1)                \n\
        stepsize = cuda.gridsize(1)         \n\
        for i in range(start, D, stepsize): \n\
            body({0},i)                     \n\
    else:                                   \n\
        for i in prange(D):                 \n\
            body({0},i)                     '.format(argstring)
    print(code)

    tempdict={'body':jittedbody}
    globals()['body']=jittedbody
    exec(code,globals(),tempdict)
    return cudaswitch_kernel(signature)(tempdict['F'])







####################################################################################################

def new_value_at_i(signature):
    def dec(body):
        return get_optionalcuda_parallelized(signature,body)
    return dec

####################################################################################################







def wrapcuda(f):
    def f_(*args):
        moved=[isinstance(x,np.ndarray) and cuda_on for x in args]
        deviceargs=[cuda.to_device(x) if m else x for m,x in zip(moved,args)]
        f(*deviceargs)
        args=[x.copy_to_host() if m else x for m,x in zip(moved,deviceargs)]
        return args
    return f_



if __name__=='__main__':
    @new_value_at_i(void(float64[:],uint64))
    def f(X,i):
        X[i]=X[i]**2


    Y=wrapcuda(f)(np.arange(10**6).astype(float),10**6)
    print(Y)



