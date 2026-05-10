import numpy as np
import strawberryfields as sf
from strawberryfields import ops

eta = 0.01
nbar = 1000
theta_bs = np.arccos(np.sqrt(eta))
num_shots = 10000

measurements_mode0 = []

for _ in range(num_shots):
    # Four-mode program:
    # q[0] = Signal, q[1] = Idler, q[2] = Env (forward), q[3] = Env (backward)
    prog = sf.Program(4)
    eng = sf.Engine("gaussian")

    with prog.context as q:
        ops.S2gate(0.5) | (q[0], q[1])
        ops.Thermal(nbar) | q[2]
        ops.BSgate(theta_bs, 0.0) | (q[0], q[2])
        ops.Rgate(np.pi) | q[0]
        ops.Thermal(nbar) | q[3]
        ops.BSgate(theta_bs, 0.0) | (q[0], q[3])

        # Bob removes the key and interferes signal with idler
        ops.Rgate(-np.pi) | q[0]
        ops.BSgate(np.pi / 4, 0.0) | (q[0], q[1])
        ops.MeasureHomodyne(0.0) | q[0]
        ops.MeasureHomodyne(0.0) | q[1]

    result = eng.run(prog)
    measurements_mode0.append(result.samples[0, 0])

mean_mode0 = float(np.mean(measurements_mode0))
print(f"Mean homodyne measurement (mode 0): {mean_mode0}")
# The sign of this mean lets Bob infer the original phase bit despite heavy noise.
