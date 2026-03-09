# Baselines — Inviscid Burgers (sinusoidal IC)

## Setup
- Domain: [0, 2π], N=128, dx=0.0491
- Time: dt=0.005, N_T=3000, BURN=100
- IC: sin(x) + 0.5*sin(2x)
- Data: kandy.numerics.solve_burgers (Rusanov, TVD minmod, TVD-RK2)
- Features: ['u', 'u_x', 'u*u_x', 'u_xx']
- Train: 60000, Test: 12000
- Rollout: SSP-RK3 with CFL substeps, 300 steps

## True equation
u_t = -1.0 * u*u_x

## Results

| Method | 1-step MSE | R² | Rollout NRMSE |
|--------|-----------|------|---------------|
| OLS | 2.649e-02 | 0.8254 | 0.1036 |
| Ridge | 2.648e-02 | 0.8254 | 0.1035 |
| LASSO | 2.648e-02 | 0.8254 | 0.1022 |
| STLSQ | 2.648e-02 | 0.8254 | 0.1029 |
| Ideal sparse (c=-1.4022) | 3.705e-02 | 0.7558 | 0.0303 |

## Discovered Equations

**OLS:**
```
u_t = +0.150731*u -0.001716*u_x -1.530567*u*u_x +0.003564*u_xx +0.000326
```

**Ridge:**
```
u_t = +0.150613*u -0.001714*u_x -1.530098*u*u_x +0.003562*u_xx +0.000326
```

**LASSO:**
```
u_t = +0.149469*u -0.001586*u_x -1.528358*u*u_x +0.003555*u_xx +0.000315
```

**STLSQ:**
```
u_t = +0.150728*u -0.001688*u_x -1.530564*u*u_x +0.003564*u_xx
```

**Ideal sparse (c=-1.4022):**
```
u_t = -1.402215*u*u_x
```

## Notes
- Feature library [u, u_x, u*u_x, u_xx] matches KANDy's Koopman lift.
- True equation: u_t = -1.0*u*u_x. Only the u*u_x coefficient should be non-zero.
- Spatial derivatives use TVD minmod (consistent with KANDy).
