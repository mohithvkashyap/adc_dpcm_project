import numpy as np
from src.utils import quantize

def pcm_encode(signal,max_error,n_levels=16):
    quantized_signal,indices_pcm=quantize(signal,max_error,n_levels)
    bits=int(np.log2(n_levels))
    bitstream=[]
    for idx in indices_pcm:
        binary=format(idx,f'0{bits}b')
        bitstream.extend([int(b) for b in binary])
    return quantized_signal,indices_pcm,bitstream