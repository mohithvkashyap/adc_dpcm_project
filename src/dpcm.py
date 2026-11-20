import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src.utils import quantize

def dpcm_encode(x,max_error,n_levels=16,a=1.0):
    n=len(x)
    x_hat=np.zeros(n)
    e=np.zeros(n)
    indices=np.zeros(n,dtype=int)
    for i in range(n):
        predicted=a*x_hat[i-1] if i>0 else 0
        error=x[i]-predicted
        e[i]=error
        quantized_error,index=quantize(error,max_error,n_levels)
        x_hat[i]=predicted+quantized_error
        indices[i]=index
    return x_hat,e,indices

def dpcm_decode(indices,max_error,n_levels=16,a=1.0):
    n=len(indices)
    x_hat=np.zeros(n)
    step=(2*max_error)/n_levels
    levels=-max_error+step*(np.arange(n_levels)+0.5)
    for i in range(n):
        predicted=a*x_hat[i-1] if i>0 else 0
        x_hat[i]=predicted+levels[indices[i]]
    return x_hat