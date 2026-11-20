import numpy as np
import matplotlib.pyplot as plt

from src.dpcm import dpcm_encode, dpcm_decode
from src.pcm import pcm_encode
from src.dm import dm_encode, dm_decode
from src.utils import (
    generate_signal, snr, prd,
    dequantize, indices_to_bits, bits_to_indices, inject_errors
)

# -----------------------------------
# Signal parameters
# -----------------------------------
F = 2000          # signal frequency (Hz)
FS = 32000        # sampling frequency (Hz)
DURATION = 0.01   # signal duration (s)
A = 5             # signal amplitude
N_LEVELS = 16     # quantization levels (4-bit) for PCM/DPCM
N_BITS = 4        # bits per sample for PCM/DPCM
P_ERROR = 0.01    # bit error probability for Phase 3

t, x = generate_signal(f=F, fs=FS, duration=DURATION, a=A)

# ===================================
# DPCM
# ===================================
e_est = x - np.roll(x, 1)
e_est[0] = x[0]
dpcm_margin = 1.2
max_error_dpcm = np.max(np.abs(e_est)) * dpcm_margin

x_hat_dpcm, e_dpcm, indices_dpcm = dpcm_encode(x, max_error_dpcm, n_levels=N_LEVELS)

print("DPCM")
print("max reconstruction error:", np.max(np.abs(x - x_hat_dpcm)))
print("mean reconstruction error:", np.mean(np.abs(x - x_hat_dpcm)))
print("DPCM SNR:", round(snr(x, x_hat_dpcm), 2), "dB")
print("DPCM PRD:", round(prd(x, x_hat_dpcm), 2), "%")
print()

# ===================================
# PCM
# ===================================
max_error_pcm = np.max(np.abs(x))
x_pcm, indices_pcm, bitstream_pcm = pcm_encode(x, max_error_pcm, n_levels=N_LEVELS)

print("PCM")
print("max reconstruction error:", np.max(np.abs(x - x_pcm)))
print("mean reconstruction error:", np.mean(np.abs(x - x_pcm)))
print("PCM SNR:", round(snr(x, x_pcm), 2), "dB")
print("PCM PRD:", round(prd(x, x_pcm), 2), "%")
print("Bit rate:", len(bitstream_pcm) / DURATION, "bits/sec")
print()

# ===================================
# DM — optimal step size (margin found via sweep below)
# ===================================
dm_margin = 0.9
step_dm = ((2 * np.pi * F * A) / FS) * dm_margin
x_hat_dm, bits_dm = dm_encode(x, step_dm)

print("DM")
print("max reconstruction error:", np.max(np.abs(x - x_hat_dm)))
print("mean reconstruction error:", np.mean(np.abs(x - x_hat_dm)))
print("DM SNR:", round(snr(x, x_hat_dm), 2), "dB")
print("DM PRD:", round(prd(x, x_hat_dm), 2), "%")
print("Bit rate:", len(bits_dm) / DURATION, "bits/sec")
print()

# DM step-size margin sweep (shows slope overload vs granular noise tradeoff)
margins = np.arange(0.5, 2.0, 0.05)
snr_vals = []
for m in margins:
    step_m = ((2 * np.pi * F * A) / FS) * m
    x_hat_m, _ = dm_encode(x, step_m)
    snr_vals.append(snr(x, x_hat_m))

best_idx = np.argmax(snr_vals)
print("Best DM margin:", round(margins[best_idx], 2))
print("Best DM SNR:", round(snr_vals[best_idx], 2), "dB")
print()

# ===================================
# Bit-error robustness (Phase 3)
# ===================================

# PCM under bit errors
pcm_bits_flat = indices_to_bits(indices_pcm, N_BITS)
pcm_bits_corrupt = inject_errors(pcm_bits_flat, P_ERROR)
indices_pcm_corrupt = bits_to_indices(pcm_bits_corrupt, N_BITS)
x_pcm_corrupt = dequantize(indices_pcm_corrupt, max_error_pcm, N_LEVELS)

print("PCM under bit errors")
print("SNR:", round(snr(x, x_pcm_corrupt), 2), "dB")
print("PRD:", round(prd(x, x_pcm_corrupt), 2), "%")
print()

# DPCM under bit errors
dpcm_bits_flat = indices_to_bits(indices_dpcm, N_BITS)
dpcm_bits_corrupt = inject_errors(dpcm_bits_flat, P_ERROR)
indices_dpcm_corrupt = bits_to_indices(dpcm_bits_corrupt, N_BITS)
x_hat_dpcm_corrupt = dpcm_decode(indices_dpcm_corrupt, max_error_dpcm, N_LEVELS)

print("DPCM under bit errors")
print("SNR:", round(snr(x, x_hat_dpcm_corrupt), 2), "dB")
print("PRD:", round(prd(x, x_hat_dpcm_corrupt), 2), "%")
print()

# DM under bit errors (bits are already binary, no conversion needed)
dm_bits_corrupt = inject_errors(bits_dm, P_ERROR)
x_hat_dm_corrupt = dm_decode(dm_bits_corrupt, step_dm)

print("DM under bit errors")
print("SNR:", round(snr(x, x_hat_dm_corrupt), 2), "dB")
print("PRD:", round(prd(x, x_hat_dm_corrupt), 2), "%")
print()

# ===================================
# Plots
# ===================================

# PCM: original, quantized, reconstructed
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 2)
plt.step(t, x_pcm, where='mid')
plt.title("PCM Quantized Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, x, label="Original")
plt.plot(t, x_pcm, '--', label="Reconstructed")
plt.title("PCM Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("pcm_plot.png", dpi=300, bbox_inches='tight')

# PCM vs DPCM reconstruction (zoomed for visibility)
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, 0.0015)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, x, label="Original")
plt.plot(t, x_pcm, '--', label="PCM Reconstructed")
plt.title("PCM Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.xlim(0, 0.0015)
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, x, label="Original")
plt.plot(t, x_hat_dpcm, '--', label="DPCM Reconstructed")
plt.title("DPCM Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.xlim(0, 0.0015)
plt.grid()

plt.tight_layout()
plt.savefig("pcm_vs_dpcm_plot.png", dpi=300, bbox_inches='tight')

# PCM vs DPCM reconstruction error
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, x - x_pcm)
plt.title("PCM Reconstruction Error")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, x - x_hat_dpcm)
plt.title("DPCM Reconstruction Error")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.grid()

plt.tight_layout()
plt.savefig("pcm_vs_dpcm_error_plot.png", dpi=300, bbox_inches='tight')

# DPCM: original vs reconstructed
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, x, color='tab:blue', linewidth=1.5)
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, x_hat_dpcm, color='tab:orange', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (DPCM)')
plt.grid(True)

plt.tight_layout()
plt.savefig('dpcm_plot_separate.png', dpi=150)

# DM margin sweep — shows slope overload vs granular noise tradeoff
plt.figure(figsize=(8, 5))
plt.plot(margins, snr_vals, marker='o')
plt.axvline(1.0, linestyle='--', color='gray', label='Theoretical no-overload threshold')
plt.xlabel("Step size margin (× theoretical minimum)")
plt.ylabel("SNR (dB)")
plt.title("DM: SNR vs Step Size Margin")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("dm_margin_sweep.png", dpi=300, bbox_inches='tight')