import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_audio(path,target_fs=8000,max_seconds=3,start_seconds=0):
    fs,x=wavfile.read(path)
    x=x.astype(np.float64)
    if x.ndim>1:
        x=x.mean(axis=1)                          # stereo -> mono
    x-=x.mean()                                   # remove DC offset
    if fs!=target_fs:
        g=np.gcd(int(fs),int(target_fs))
        x=resample_poly(x,target_fs//g,fs//g)
    start=int(start_seconds*target_fs)
    x=x[start:start+int(max_seconds*target_fs)]
    return target_fs,x/np.max(np.abs(x))          # normalise to +-1

def seg_snr(x,x_hat,frame=256):
    n=len(x)//frame*frame
    xs=x[:n].reshape(-1,frame)
    es=(x[:n]-x_hat[:n]).reshape(-1,frame)
    s=10*np.log10((np.sum(xs**2,axis=1)+1e-12)/(np.sum(es**2,axis=1)+1e-12))
    return np.mean(np.clip(s,-10,35))             # standard clamp range

def save_wav(path,fs,x,out_fs=44100):
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    if fs!=out_fs:                                # 8 kHz files don't play in some players
        g=np.gcd(int(fs),int(out_fs))
        x=resample_poly(x,out_fs//g,fs//g)
    wavfile.write(path,out_fs,(np.clip(x,-1,1)*32767).astype(np.int16))