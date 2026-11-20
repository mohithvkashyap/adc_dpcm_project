import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.signal import resample_poly

from src.pcm import pcm_encode
from src.dpcm import dpcm_encode, dpcm_decode
from src.dm import dm_encode, dm_decode
from src.utils import snr, prd, dequantize, indices_to_bits, bits_to_indices, inject_errors
from src.audio import load_audio, seg_snr, save_wav

# -----------------------------------
# Settings
# -----------------------------------
AUDIO_FILE = "audio/sample.wav"
TARGET_FS = 8000      # audio is resampled to this rate (8 kHz = telephone quality)
MAX_SECONDS = 3       # clip length (the sample-by-sample loops get slow on long clips)
N_BITS = 4
N_LEVELS = 2 ** N_BITS
OSR = N_BITS          # DM runs OSR x faster, so all three methods use the same bit rate
P_ERROR = 0.01
OUT = "audio_out/"

def warn_edge(name, i, n):
    if i == n - 1:
        print(f"WARNING: best {name} is at the end of its sweep range - widen the range")

np.random.seed(0)     # same bit errors every run
fs, x = load_audio(AUDIO_FILE, TARGET_FS, MAX_SECONDS)
print(f"Loaded {len(x)/fs:.2f} s at {fs} Hz ({len(x)} samples)")
print(f"Bit rate (all methods): {fs*N_BITS} bits/sec\n")

# ===================================
# PCM — tune the quantizer range (clip rare peaks, finer steps elsewhere)
# ===================================
pcm_scales = np.arange(0.1, 1.01, 0.05)
pcm_snr = [snr(x, pcm_encode(x, s, N_LEVELS)[0]) for s in pcm_scales]
max_error_pcm = pcm_scales[np.argmax(pcm_snr)]
x_pcm, indices_pcm, _ = pcm_encode(x, max_error_pcm, N_LEVELS)

# ===================================
# DPCM — leaky first-order predictor + tuned error range
# ===================================
a = np.dot(x[1:], x[:-1]) / np.dot(x[:-1], x[:-1])      # optimal 1st-order predictor coefficient
peak_e = np.max(np.abs(x[1:] - a * x[:-1]))
dpcm_scales = np.arange(0.05, 1.31, 0.05)
dpcm_snr = [snr(x, dpcm_encode(x, s * peak_e, N_LEVELS, a)[0]) for s in dpcm_scales]
max_error_dpcm = dpcm_scales[np.argmax(dpcm_snr)] * peak_e
warn_edge("DPCM error range", np.argmax(dpcm_snr), len(dpcm_scales))
x_dpcm, _, indices_dpcm = dpcm_encode(x, max_error_dpcm, N_LEVELS, a)
print("Predictor coefficient a:", round(a, 3))
print("DPCM encoder/decoder match:",
      np.allclose(x_dpcm, dpcm_decode(indices_dpcm, max_error_dpcm, N_LEVELS, a)))

# ===================================
# DM — runs at OSR x the sample rate, leaky integrator (keeps bit errors from drifting forever),
#      decoder output is low-pass filtered and decimated back to the base rate
# ===================================
x_up = resample_poly(x, OSR, 1)
base_step = np.mean(np.abs(np.diff(x_up)))

def dm_run(step, leak):
    x_hat_up, bits = dm_encode(x_up, step, leak)
    return resample_poly(x_hat_up, 1, OSR), bits

dm_leaks = [1.0, 0.99, 0.98, 0.97, 0.95, 0.92, 0.9]
dm_scales = np.geomspace(0.5, 64, 25)
dm_snr = np.array([[snr(x, dm_run(s * base_step, lk)[0]) for s in dm_scales] for lk in dm_leaks])
li, si = np.unravel_index(np.argmax(dm_snr), dm_snr.shape)
leak_dm, step_dm = dm_leaks[li], dm_scales[si] * base_step
warn_edge("DM step size", si, len(dm_scales))
warn_edge("DM leak", li, len(dm_leaks))
print(f"DM step = {dm_scales[si]:.1f} x mean|dx|, leak = {leak_dm}")
x_dm, bits_dm = dm_run(step_dm, leak_dm)

# ===================================
# Bit errors
# ===================================
bits = inject_errors(indices_to_bits(indices_pcm, N_BITS), P_ERROR)
x_pcm_err = dequantize(bits_to_indices(bits, N_BITS), max_error_pcm, N_LEVELS)

bits = inject_errors(indices_to_bits(indices_dpcm, N_BITS), P_ERROR)
x_dpcm_err = dpcm_decode(bits_to_indices(bits, N_BITS), max_error_dpcm, N_LEVELS, a)

x_dm_err = resample_poly(dm_decode(inject_errors(bits_dm, P_ERROR), step_dm, leak_dm), 1, OSR)

# ===================================
# Results
# ===================================
results = {"PCM": (x_pcm, x_pcm_err), "DPCM": (x_dpcm, x_dpcm_err), "DM": (x_dm, x_dm_err)}

print(f"\n{'':6}{'--- clean channel ---':>31}{'--- p=' + str(P_ERROR) + ' bit errors ---':>34}")
print(f"{'':6}{'SNR':>8}{'SegSNR':>9}{'PRD%':>8}{'':4}{'SNR':>8}{'SegSNR':>9}{'PRD%':>8}")
for name, (clean, err) in results.items():
    print(f"{name:6}{snr(x, clean):8.2f}{seg_snr(x, clean):9.2f}{prd(x, clean):8.2f}{'':4}"
          f"{snr(x, err):8.2f}{seg_snr(x, err):9.2f}{prd(x, err):8.2f}")

# listen to everything
save_wav(OUT + "original.wav", fs, x)
for name, (clean, err) in results.items():
    save_wav(OUT + f"{name.lower()}.wav", fs, clean)
    save_wav(OUT + f"{name.lower()}_bit_errors.wav", fs, err)

# ===================================
# Plots
# ===================================

# tuning sweeps
plt.figure(figsize=(12, 4))
for i, (xs, ys, title, label) in enumerate([
        (pcm_scales, pcm_snr, "PCM", "Quantizer range (× peak)"),
        (dpcm_scales, dpcm_snr, "DPCM", "Error range (× open-loop peak)")]):
    plt.subplot(1, 3, i + 1)
    plt.plot(xs, ys, marker='o')
    plt.axvline(xs[np.argmax(ys)], linestyle='--', color='gray', label='Best')
    plt.title(f"{title}: SNR vs range")
    plt.xlabel(label)
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid()

plt.subplot(1, 3, 3)
for lk, row in zip(dm_leaks, dm_snr):
    plt.plot(dm_scales, row, label=f"leak {lk}")
plt.axvline(dm_scales[si], linestyle='--', color='gray')
plt.xscale('log')
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.title("DM: SNR vs step size")
plt.xlabel("Step size (× mean |Δx|)")
plt.ylabel("SNR (dB)")
plt.legend(fontsize=7)
plt.grid()
plt.tight_layout()
plt.savefig("audio_tuning.png", dpi=300, bbox_inches='tight')

# reconstruction around the loudest part of the clip (30 ms window)
t = np.arange(len(x)) / fs
w = int(0.03 * fs)
lo = max(np.argmax(np.abs(x)) - w // 2, 0)
sl = slice(lo, lo + w)
plt.figure(figsize=(10, 9))
for i, (name, (clean, _)) in enumerate(results.items()):
    plt.subplot(3, 1, i + 1)
    plt.plot(t[sl], x[sl], label="Original")
    plt.plot(t[sl], clean[sl], '--', label=f"{name} reconstructed")
    plt.title(f"{name} Reconstruction (30 ms around loudest part)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid()
plt.tight_layout()
plt.savefig("audio_comparison.png", dpi=300, bbox_inches='tight')

# summary: clean vs bit errors
idx = np.arange(len(results))
plt.figure(figsize=(7, 5))
plt.bar(idx - 0.2, [snr(x, r[0]) for r in results.values()], 0.4, label="Clean channel")
plt.bar(idx + 0.2, [snr(x, r[1]) for r in results.values()], 0.4, label=f"Bit errors (p={P_ERROR})")
plt.xticks(idx, results.keys())
plt.ylabel("SNR (dB)")
plt.title("Audio: PCM vs DPCM vs DM")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("audio_summary.png", dpi=300, bbox_inches='tight')