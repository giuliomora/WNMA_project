from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .config import ClassicalConfig


def run_classical(cfg: ClassicalConfig, output_dir: Path, log) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, cfg.time_end, cfg.N_samples)
    signal_tx = np.sqrt(cfg.N_S) * np.sin(2.0 * np.pi * cfg.carrier_freq * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal_tx)
    plt.title("Clean Sine Wave Carrier (Ultra-Low Power)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_dir / "clean_sine_wave.png")

    noise_forward = (
        np.random.normal(scale=np.sqrt(cfg.N_B / 2.0), size=cfg.N_samples)
        + 1j * np.random.normal(scale=np.sqrt(cfg.N_B / 2.0), size=cfg.N_samples)
    )
    signal_alice_rx = np.sqrt(cfg.eta) * signal_tx + np.sqrt(1.0 - cfg.eta) * noise_forward

    zoom_end = min(cfg.plot_zoom_samples, cfg.N_samples)
    plt.figure(figsize=(12, 5))
    plt.plot(t[:zoom_end], signal_tx[:zoom_end], label="signal_tx (clean)", color="blue", linewidth=2)
    plt.plot(
        t[:zoom_end],
        np.real(signal_alice_rx[:zoom_end]),
        label="signal_alice_rx (noisy)",
        color="red",
        alpha=0.7,
        linewidth=1,
    )
    plt.title("Forward Channel: Signal Buried Under Thermal Noise (Zoom)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "forward_channel_noise.png")

    phi = float(np.random.choice([0.0, np.pi]))
    phi_k = float(np.random.choice([0.0, np.pi]))
    theta = phi + phi_k
    signal_alice_tx = signal_alice_rx * np.exp(-1j * theta)

    log(f"phi (info bit): {phi}")
    log(f"phi_k (key bit): {phi_k}")

    noise_backward = (
        np.random.normal(scale=np.sqrt(cfg.N_B / 2.0), size=cfg.N_samples)
        + 1j * np.random.normal(scale=np.sqrt(cfg.N_B / 2.0), size=cfg.N_samples)
    )
    signal_bob_rx = np.sqrt(cfg.eta) * signal_alice_tx + np.sqrt(1.0 - cfg.eta) * noise_backward

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(signal_bob_rx), np.imag(signal_bob_rx), s=8, alpha=0.4)
    plt.title("Backward Channel: IQ Scatter (Constellation Destroyed)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "backward_channel_iq_scatter.png")

    signal_bob_unmasked = signal_bob_rx * np.exp(1j * phi_k)
    decision_metric = np.sum(np.real(signal_bob_unmasked))
    phi_hat = 0.0 if decision_metric > 0.0 else np.pi

    log(f"phi_hat (estimated bit): {phi_hat}")
    log("Decoding successful" if phi_hat == phi else "Decoding error")

    eve_corr = float(np.corrcoef(np.real(signal_alice_rx), np.real(signal_bob_rx))[0, 1])
    log(f"Eve correlation (real parts): {eve_corr}")

    N_S_array = np.logspace(
        np.log10(cfg.N_S_min),
        np.log10(cfg.N_S_max),
        cfg.N_S_points,
    )
    beta_classical = N_S_array / cfg.N_B
    beta_quantum = 4.0 * (N_S_array / cfg.N_B)

    P_err_classical = 0.5 * np.exp(-cfg.chernoff_M * (cfg.eta**2) * beta_classical)
    P_err_quantum = 0.5 * np.exp(-cfg.chernoff_M * (cfg.eta**2) * beta_quantum)

    plt.figure(figsize=(10, 5))
    plt.semilogy(N_S_array, P_err_classical, label="Classical (coherent)")
    plt.semilogy(N_S_array, P_err_quantum, label="Quantum (TMSV)")
    plt.title("Theoretical Error Probabilities: Classical vs Quantum")
    plt.xlabel("Signal photon number N_S")
    plt.ylabel("Error probability P_err")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "quantum_advantage_bounds.png")

    return {
        "phi": phi,
        "phi_k": phi_k,
        "phi_hat": phi_hat,
        "eve_corr": eve_corr,
    }
