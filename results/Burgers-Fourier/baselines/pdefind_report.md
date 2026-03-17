# PDE-FIND + KANDy Comparison — Inviscid Burgers (Fourier IC)

## Setup
- Domain: [-3.1416, 3.1416], Nx=128, dx=0.04908738521234052
- Time: [0, 2.0], dt=0.004, Nt=501
- IC: 10 Fourier modes, power-law decay p=1.5, seed=0
- Solver: Rusanov flux + RK45 (rtol=1e-6, atol=1e-8)
- Rollout: SSP-RK3 with CFL substeps (CFL=0.35), 501 steps

## True equation
u_t = -u * u_x   (inviscid Burgers)

## Results

| Method | Terms | Rollout NRMSE | Equation |
|--------|-------|---------------|----------|
| PDE-FIND FD deg=2 (thr=0.05) | 1 | 1.5615 | `u_t = +12.272*x0_1` |
| PDE-FIND SmoothedFD deg=2 (thr=0.5) | 1 | 1.5615 | `u_t = +12.272*x0_1` |
| PDE-FIND SavGol deg=2 (thr=0.05) | 12 | N/A | `u_t = +4.052*1 -4.574*x0 -7.08*x0^2 +4.922*x0_1 -0.083*x0_11 +0.011*x0_111 +3.072*x0x0_1 +1.153*x0^2x0_1 -0.184*x0x0_11 -0.052*x0^2x0_11` |
| OLS | 5 | 0.1497 | `u_t = +0.063*u -0.082*u_x -1.083*u*u_x +0.02*u_xx +0.024` |
| LASSO (alpha=0.001) | 5 | 0.1497 | `u_t = +0.062*u -0.082*u_x -1.081*u*u_x +0.02*u_xx +0.024` |
| KANDy [3,1] | — | 0.0486 | `u_t = -1.084*u*u_x - 0.026` |

## PDE-FIND Library Terms

### PDE-FIND FD deg=2 (threshold=0.05)

| Term | Coefficient | Active |
|------|------------|--------|
| 1 | +0.000000 |  |
| x0 | +0.000000 |  |
| x0^2 | -0.000000 |  |
| x0_1 | +12.271846 | yes |
| x0_11 | -0.000000 |  |
| x0_111 | +0.000000 |  |
| x0x0_1 | +0.000000 |  |
| x0^2x0_1 | +0.000000 |  |
| x0x0_11 | -0.000000 |  |
| x0^2x0_11 | +0.000000 |  |
| x0x0_111 | -0.000000 |  |
| x0^2x0_111 | +0.000000 |  |

### PDE-FIND SmoothedFD deg=2 (threshold=0.5)

| Term | Coefficient | Active |
|------|------------|--------|
| 1 | +0.000000 |  |
| x0 | +0.000000 |  |
| x0^2 | +0.000000 |  |
| x0_1 | +12.271846 | yes |
| x0_11 | +0.000000 |  |
| x0_111 | +0.000000 |  |
| x0x0_1 | +0.000000 |  |
| x0^2x0_1 | +0.000000 |  |
| x0x0_11 | +0.000000 |  |
| x0^2x0_11 | +0.000000 |  |
| x0x0_111 | +0.000000 |  |
| x0^2x0_111 | +0.000000 |  |

### PDE-FIND SavGol deg=2 (threshold=0.05)

| Term | Coefficient | Active |
|------|------------|--------|
| 1 | +4.051748 | yes |
| x0 | -4.574411 | yes |
| x0^2 | -7.080266 | yes |
| x0_1 | +4.921993 | yes |
| x0_11 | -0.082605 | yes |
| x0_111 | +0.010704 | yes |
| x0x0_1 | +3.072352 | yes |
| x0^2x0_1 | +1.152979 | yes |
| x0x0_11 | -0.184269 | yes |
| x0^2x0_11 | -0.051528 | yes |
| x0x0_111 | +0.000897 | yes |
| x0^2x0_111 | -0.003750 | yes |

## Notes
- PDE-FIND uses PySINDy's PDELibrary (finite difference derivatives) + STLSQ optimizer
- KANDy uses TVD minmod derivatives with conservation-form feature d(u²/2)/dx
- OLS/LASSO use hand-crafted library [u, u_x, u*u_x, u_xx] with TVD minmod derivatives
- True equation: u_t = -1.0*u*u_x (single active term)
