# Baseline Comparison -- Inviscid Burgers with Random Fourier ICs

## Setup (matches KANDy experiment exactly)
- Domain: [-3.141592653589793, 3.141592653589793], Nx=128, dx=0.04908738521234052
- Time: [0, 2.0], dt_data=0.004, Nt=501
- IC: 10 random Fourier modes, power-law decay p=1.5
- Ground truth: Rusanov flux + RK45 (rtol=1e-7, atol=1e-9)
- Spatial derivatives: TVD minmod (u_x), standard Laplacian (u_xx)
- Rollout: SSP-RK3 with CFL-adaptive substeps (CFL=0.35)

## True equation
u_t = -1.0*u*u_x   (inviscid Burgers)

## Results

| Method | 1-step MSE | R2 | NRMSE (t<1) | NRMSE (full) |
|--------|-----------|------|-------------|--------------|
| OLS [u, u_x, u*u_x, u_xx] | 1.652e-01 | 0.8527 | 0.1033 | 0.1509 |
| Ridge(alpha=10.0) [u,u_x,u*u_x,u_xx] | 1.652e-01 | 0.8527 | 0.1034 | 0.1509 |
| LASSO(alpha=0.001) [u,u_x,u*u_x,u_xx] | 1.652e-01 | 0.8527 | 0.1037 | 0.1509 |
| PDE-FIND STLSQ(thr=0.005,deg=3) | 1.310e-01 | 0.8832 | 0.0908 | 0.0817 |
| PDE-FIND SR3(lam=0.1,deg=2) | 1.122e+00 | -0.0000 | 0.4503 | 0.6352 |
| PDE-FIND tuned STLSQ(thr=0.001) | 1.656e-01 | 0.8524 | 0.1017 | 0.1468 |
| Ideal sparse (u_t = -0.5255*u*u_x) | 6.366e-01 | 0.4324 | 0.3824 | 0.5785 |

## Discovered Equations

**OLS:**
```
u_t = +0.062850*u -0.083972*u_x -1.080137*u*u_x +0.019702*u_xx +0.025990
```

**Ridge:**
```
u_t = +0.062757*u -0.083952*u_x -1.079852*u*u_x +0.019697*u_xx +0.025984
```

**LASSO:**
```
u_t = +0.061438*u -0.083551*u_x -1.078347*u*u_x +0.019664*u_xx +0.025890
```

**PDE-FIND STLSQ:**
```
u_t = +0.072636*1 +0.218360*u -0.109634*u_x +0.017661*u_xx -0.046644*u*u -1.163423*u*u_x -0.066927*u*u*u +0.063812*u*u*u_x -0.019450*u*u_x*u_x
```

**PDE-FIND SR3:**
```
u_t = 0
```

**Tuned STLSQ:**
```
u_t = +0.062644*u -0.082250*u_x -1.079546*u*u_x +0.019691*u_xx
```

**Ideal sparse:**
```
u_t = -0.525515*u*u_x
```

## PDE-FIND best configuration
- STLSQ: {'degree': 3, 'threshold': 0.005, 'alpha': 0.01}
- SR3: {'degree': 2, 'lam': 0.1, 'nu': 10.0}

## Notes
- The hand-crafted library [u, u_x, u*u_x, u_xx] is the SAME lift used by KANDy.
  These baselines show what linear regression on that lift achieves.
- PDE-FIND uses a larger polynomial library (degree 2-3 monomials of u, u_x, u_xx)
  and STLSQ sparsification, which is the standard SINDy/PDE-FIND pipeline.
- The Fourier IC creates complex shock interactions that stress all methods.
