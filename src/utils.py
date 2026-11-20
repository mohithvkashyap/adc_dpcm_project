import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def generate_signal(f=2000,fs=32000,duration=0.01,a=5):
    t=np.linspace(0,duration,int(fs*duration))
    signal=a*np.sin(2*np.pi*f*t)
    return t,signal

def quantize(e,max_error,n_levels=16):
    step=(2*max_error)/n_levels
    levels=-max_error+step*(np.arange(n_levels)+0.5)
    indices=np.round((e+max_error)/step-0.5).astype(int)
    indices=np.clip(indices,0,n_levels-1)
    quantized_signal=levels[indices]
    return quantized_signal,indices

def snr(x,x_hat):
    signal_power=np.mean(x**2)
    noise_power=np.mean((x-x_hat)**2)
    return 10*np.log10(signal_power/noise_power)

def prd(x,x_hat):
    return np.sqrt(np.sum((x-x_hat)**2))/np.sqrt(np.sum(x**2))*100

def dequantize(indices,max_error,n_levels=16):
    step=(2*max_error)/n_levels
    levels=-max_error+step*(np.arange(n_levels)+0.5)
    return levels[indices]

def indices_to_bits(indices,n_bits):
    bits=[]
    for idx in indices:
        bits.extend([int(b) for b in format(idx,f'0{n_bits}b')])
    return np.array(bits)

def bits_to_indices(bits,n_bits):
    bits=np.array(bits).reshape(-1,n_bits)
    return np.array([int(''.join(map(str,row)),2) for row in bits])

def inject_errors(bits,p):
    bits=np.array(bits).copy()
    flip=np.random.rand(len(bits))<p
    bits[flip]=1-bits[flip]
    return bits