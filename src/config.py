import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ClassicalConfig:
    N_samples: int
    N_S: float
    N_B: float
    eta: float
    carrier_freq: float
    time_end: float
    plot_zoom_samples: int
    chernoff_M: int
    N_S_min: float
    N_S_max: float
    N_S_points: int


@dataclass
class QuantumConfig:
    eta: float
    nbar: float
    s2_r: float
    phi: Any
    phi_k: Any
    shots: int


@dataclass
class SimConfig:
    output_dir: str
    seed: int
    classical: ClassicalConfig
    quantum: QuantumConfig


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Path) -> SimConfig:
    raw = _load_json(path)
    classical = raw["classical"]
    quantum = raw["quantum"]

    return SimConfig(
        output_dir=raw["output_dir"],
        seed=raw["seed"],
        classical=ClassicalConfig(**classical),
        quantum=QuantumConfig(**quantum),
    )
