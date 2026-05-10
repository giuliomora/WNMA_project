import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
N_samples = 1000
N_S = 0.01
N_B = 1000
eta = 0.01

# Time array
t = np.linspace(0.0, 100.0, N_samples)

# Clean sine wave carrier with ultra-low amplitude
carrier_freq = 1.0  # Hz
signal_tx = np.sqrt(N_S) * np.sin(2.0 * np.pi * carrier_freq * t)

# Plot the clean signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal_tx)
plt.title("Clean Sine Wave Carrier (Ultra-Low Power)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.tight_layout()
# Salva il plot nella cartella output
plt.savefig("output/clean_sine_wave.png")

# Phase 2: Forward channel towards Alice
# Complex AWGN with variance N_B
noise_forward = (
	np.random.normal(scale=np.sqrt(N_B / 2.0), size=N_samples)
	+ 1j * np.random.normal(scale=np.sqrt(N_B / 2.0), size=N_samples)
)
signal_alice_rx = np.sqrt(eta) * signal_tx + np.sqrt(1.0 - eta) * noise_forward

# Plot Tx vs. Rx (real part) to show the signal buried in noise
# Zoom su una finestra temporale ristretta per maggiore leggibilità
zoom_start = 0
zoom_end = 200  # primi 200 campioni

plt.figure(figsize=(12, 5))
plt.plot(t[zoom_start:zoom_end], signal_tx[zoom_start:zoom_end], label="signal_tx (pulito)", color="blue", linewidth=2)
plt.plot(t[zoom_start:zoom_end], np.real(signal_alice_rx[zoom_start:zoom_end]), label="signal_alice_rx (con rumore)", color="red", alpha=0.7, linewidth=1)
plt.title("Forward Channel: Segnale sommerso dal rumore termico (zoom)")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("output/forward_channel_noise.png")

# Phase 3: Secret Message and Passive Modulation (Alice)
phi = np.random.choice([0.0, np.pi])
phi_k = np.random.choice([0.0, np.pi])
theta = phi + phi_k
signal_alice_tx = signal_alice_rx * np.exp(-1j * theta)

print(f"phi (info bit): {phi}")
print(f"phi_k (key bit): {phi_k}")

# Phase 4: Backward channel towards Bob
noise_backward = (
	np.random.normal(scale=np.sqrt(N_B / 2.0), size=N_samples)
	+ 1j * np.random.normal(scale=np.sqrt(N_B / 2.0), size=N_samples)
)
signal_bob_rx = np.sqrt(eta) * signal_alice_tx + np.sqrt(1.0 - eta) * noise_backward

# IQ scatter plot to show constellation collapse
plt.figure(figsize=(6, 6))
plt.scatter(np.real(signal_bob_rx), np.imag(signal_bob_rx), s=8, alpha=0.4)
plt.title("Backward Channel: IQ Scatter (Constellation Destroyed)")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.axis("equal")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("output/backward_channel_iq_scatter.png")

# Phase 5: Classical Demodulation and BER (Bob)
signal_bob_unmasked = signal_bob_rx * np.exp(1j * phi_k)
decision_metric = np.sum(np.real(signal_bob_unmasked))
phi_hat = 0.0 if decision_metric > 0.0 else np.pi

print(f"phi_hat (estimated bit): {phi_hat}")
print("Decoding successful" if phi_hat == phi else "Decoding error")

# Phase 6: Interceptor / Covertness Proof (Eve)
eve_corr = np.corrcoef(np.real(signal_alice_rx), np.real(signal_bob_rx))[0, 1]
print(f"Eve correlation (real parts): {eve_corr}")

# Phase 7: Analytical Quantum Advantage (Chernoff Bounds)
N_S_array = np.logspace(-3, 0, 200)
N_B_theory = 1000.0
M = 1_000_000

beta_classical = N_S_array / N_B_theory
beta_quantum = 4.0 * (N_S_array / N_B_theory)

P_err_classical = 0.5 * np.exp(-M * (eta**2) * beta_classical)
P_err_quantum = 0.5 * np.exp(-M * (eta**2) * beta_quantum)

plt.figure(figsize=(10, 5))
plt.semilogy(N_S_array, P_err_classical, label="Classical (coherent)")
plt.semilogy(N_S_array, P_err_quantum, label="Quantum (TMSV)")
plt.title("Theoretical Error Probabilities: Classical vs Quantum")
plt.xlabel("Signal photon number N_S")
plt.ylabel("Error probability P_err")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("output/quantum_advantage_bounds.png")
