import numpy as np

def dm_encode(x,step,leak=1.0):
    n=len(x)
    x_hat=np.zeros(n)
    bits=np.zeros(n,dtype=int)
    for i in range(n):
        predicted=leak*x_hat[i-1] if i>0 else 0
        if x[i]>=predicted:
            x_hat[i]=predicted+step
            bits[i]=1
        else:
            x_hat[i]=predicted-step
            bits[i]=0
    return x_hat,bits

def dm_decode(bits,step,leak=1.0):
    n=len(bits)
    x_hat=np.zeros(n)
    for i in range(n):
        predicted=leak*x_hat[i-1] if i>0 else 0
        x_hat[i]=predicted+step if bits[i]==1 else predicted-step
    return x_hat