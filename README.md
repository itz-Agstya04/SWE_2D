# 2D Shallow Water Equations — Circular Dam Break PINN

A Physics-Informed Neural Network (PINN) surrogate for solving the 2D Shallow Water Equations (SWE) on a circular dam-break problem. The network learns to predict water depth and momentum fields across space and time by minimizing PDE residuals alongside initial and boundary condition losses.

---

## What it does

Given a square 2D domain with a circular region of deep water (the "dam"), this solver:

- Trains a neural network to satisfy three coupled PDEs simultaneously: continuity, x-momentum, and y-momentum
- Compares predictions against a classical 2D Lax-Friedrichs finite-volume reference solver
- Produces radially symmetric wave propagation — the same mathematical structure as a tropical storm or pressure system radiating outward
- Runs inference in milliseconds after training, versus seconds for the reference solver

---

## Governing equations

**Continuity:**

$$\frac{\partial h}{\partial t} + \frac{\partial q_x}{\partial x} + \frac{\partial q_y}{\partial y} = 0$$

**x-Momentum:**

$$\frac{\partial q_x}{\partial t} + \frac{\partial (u q_x)}{\partial x} + \frac{\partial (v q_x)}{\partial y} + gh\frac{\partial h}{\partial x} = 0$$

**y-Momentum:**

$$\frac{\partial q_y}{\partial t} + \frac{\partial (u q_y)}{\partial x} + \frac{\partial (v q_y)}{\partial y} + gh\frac{\partial h}{\partial y} = 0$$

Where `h` is water depth, `qx = h·u` and `qy = h·v` are momentum components, and `g = 9.81 m/s²`.

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (tested on RTX 4050 6 GB)
- PyTorch with CUDA 12.1
- DeepXDE
- NumPy, Matplotlib, Pillow

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install deepxde matplotlib numpy pillow
```

---

## Usage

```bash
python swe2d_circular_dam.py
```

The script runs end-to-end: installs packages, sets up the problem, runs the reference solver, trains the PINN in two Adam phases followed by L-BFGS, runs inference, and saves all outputs.

---

## Configuration

All tunable parameters are at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `DAM_RADIUS` | 1.5 m | Radius of the circular dam |
| `H_INSIDE` | 4.0 m | Water depth inside dam |
| `H_OUTSIDE` | 1.0 m | Water depth outside dam |
| `T_FINAL` | 2.0 s | Simulation end time |
| `NUM_DOMAIN` | 8000 | Interior collocation points |
| `NUM_BOUNDARY` | 1000 | Boundary condition points |
| `NUM_INITIAL` | 1500 | Initial condition points |
| `ADAM_ITER_PHASE1` | 8000 | Phase 1 Adam iterations (IC-heavy) |
| `ADAM_ITER_PHASE2` | 6000 | Phase 2 Adam iterations (PDE-heavy) |
| `NU_ART` | 0.002 | Artificial viscosity coefficient |

---

## Network architecture

- Input: `(x, y, t)` — 3 values
- Hidden: 6 layers × 64 neurons, tanh activation, Glorot normal initialisation
- Output: `(h, qx, qy)` — 3 values
- Input normalisation: all inputs scaled to `[-1, 1]`
- Output constraint: `h` passed through softplus to enforce positivity

The original notebook used `[80]×8` — reduced to `[64]×6` for RTX 4050 (6 GB VRAM).

---

## Training strategy

Two-phase Adam followed by L-BFGS:

**Phase 1** — IC-heavy: `W_IC = 15.0`, `W_PDE = 1.0`, `lr = 1e-3`  
The network first learns to reproduce the initial step function before enforcing physics.

**Phase 2** — PDE-heavy: `W_PDE = 3.0`, `W_IC = 5.0`, `lr = 3e-4`  
Physics residuals are upweighted to enforce the equations throughout the domain.

**L-BFGS** — fine-tunes to convergence.

A positivity penalty (`W_POS = 200.0`) with `relu(-h)` is included as a fourth PDE residual to prevent negative water depth.

Model checkpoints save every 3000 iterations to `models_2d/`.

---

## Outputs

All outputs are written to `results_2d/`:

| File | Description |
|---|---|
| `initial_condition.png` | Initial water depth field with dam boundary |
| `reference_snapshots.png` | Reference solver solution at 5 time steps |
| `loss_curve.png` | Training loss convergence |
| `comparison_grid.png` | Reference vs PINN vs absolute error at 5 times |
| `loss_and_conservation.png` | Loss curve + mass drift comparison |
| `velocity_field.png` | Depth + velocity vector field at t = 1.0s |
| `metrics.txt` | L2 error, L-inf error, mass drift, speedup |

---

## Expected performance on RTX 4050

| Stage | Time |
|---|---|
| Reference solver (80×80 grid) | ~1–2 min |
| Training (Phase 1 + 2 + L-BFGS) | ~20–35 min |
| Inference (all space-time points) | < 1 s |

---

## Connection to weather modelling

The circular dam break produces a radially symmetric outward wave — mathematically identical to a tropical storm or pressure system radiating waves. The variable mapping is direct:

| This model | Atmospheric equivalent |
|---|---|
| `h(x,y,t)` — water depth | `p(x,y,t)` — pressure |
| `u(x,y,t)` — x-velocity | `u(x,y,t)` — zonal wind |
| `v(x,y,t)` — y-velocity | `v(x,y,t)` — meridional wind |
| Pressure gradient term | Pressure gradient force |

Adding a Coriolis term to the momentum equations (`-f·qy` in x, `+f·qx` in y, where `f = 2Ω sin(lat)`) converts the radial wave into a rotating vortex — the basis of atmospheric vortex dynamics.

---

## Project structure

```
.
├── swe2d_circular_dam.py
├── README.md
├── results_2d/
│   ├── initial_condition.png
│   ├── reference_snapshots.png
│   ├── loss_curve.png
│   ├── comparison_grid.png
│   ├── loss_and_conservation.png
│   ├── velocity_field.png
│   └── metrics.txt
└── models_2d/
    ├── ckpt-*.pt
    └── swe2d_circular.*
```
