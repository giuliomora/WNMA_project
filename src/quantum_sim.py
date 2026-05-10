from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import strawberryfields as sf
from strawberryfields import ops

from .config import QuantumConfig


def _resolve_phase(value) -> float:
    if isinstance(value, str) and value.lower() == "random":
        return float(np.random.choice([0.0, np.pi]))
    return float(value)


def _build_channel_program(
    eta: float,
    nbar: float,
    s2_r: float,
    phi: float,
    phi_k: float,
) -> Tuple[sf.Program, float]:
    theta_bs = np.arccos(np.sqrt(eta))
    prog = sf.Program(4)

    with prog.context as q:
        ops.S2gate(s2_r) | (q[0], q[1])
        ops.Thermal(nbar) | q[2]
        ops.BSgate(theta_bs, 0.0) | (q[0], q[2])
        ops.Rgate(phi + phi_k) | q[0]
        ops.Thermal(nbar) | q[3]
        ops.BSgate(theta_bs, 0.0) | (q[0], q[3])

    return prog, theta_bs


def run_quantum(cfg: QuantumConfig, output_dir: Path, log) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    phi = _resolve_phase(cfg.phi)
    phi_k = _resolve_phase(cfg.phi_k)

    log(f"quantum phi (info bit): {phi}")
    log(f"quantum phi_k (key bit): {phi_k}")

    prog, theta_bs = _build_channel_program(cfg.eta, cfg.nbar, cfg.s2_r, phi, phi_k)
    eng = sf.Engine("gaussian")
    state = eng.run(prog).state
    _, cov = state.reduced_gaussian([0, 1])

    log("Covariance matrix (signal + idler):")
    log(str(cov))

    measurements_mode0 = []

    for _ in range(cfg.shots):
        prog = sf.Program(4)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.S2gate(cfg.s2_r) | (q[0], q[1])
            ops.Thermal(cfg.nbar) | q[2]
            ops.BSgate(theta_bs, 0.0) | (q[0], q[2])
            ops.Rgate(phi + phi_k) | q[0]
            ops.Thermal(cfg.nbar) | q[3]
            ops.BSgate(theta_bs, 0.0) | (q[0], q[3])

            ops.Rgate(-phi_k) | q[0]
            ops.BSgate(np.pi / 4, 0.0) | (q[0], q[1])
            ops.MeasureHomodyne(0.0) | q[0]
            ops.MeasureHomodyne(0.0) | q[1]

        result = eng.run(prog)
        measurements_mode0.append(result.samples[0, 0])

    mean_mode0 = float(np.mean(measurements_mode0))
    log(f"Mean homodyne measurement (mode 0): {mean_mode0}")

    return {
        "phi": phi,
        "phi_k": phi_k,
        "mean_mode0": mean_mode0,
    }
