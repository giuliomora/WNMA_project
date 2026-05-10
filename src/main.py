from pathlib import Path
import sys

import numpy as np

from .classical_sim import run_classical
from .config import load_config
from .quantum_sim import run_quantum


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str):
        line = str(message)
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


def main():
    config_path = Path("config/sim_config.json")
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])

    cfg = load_config(config_path)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    log = _make_logger(output_dir / "sim_log.txt")

    log("=== Classical simulation ===")
    run_classical(cfg.classical, output_dir, log)

    log("=== Quantum simulation ===")
    run_quantum(cfg.quantum, output_dir, log)


if __name__ == "__main__":
    main()
