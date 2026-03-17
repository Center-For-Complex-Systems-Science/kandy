---
name: Symbolic extraction procedure for KANDy
description: Step-by-step procedure for extracting symbolic equations from trained KANDy models using robust_auto_symbolic, including device handling, save_act, and rounding
type: feedback
---

When extracting symbolic equations from a trained KANDy model, follow this exact procedure:

```python
import copy, sympy as sp
from kandy.symbolic import robust_auto_symbolic

# 1. Deep copy the model (fix_symbolic destroys learned splines)
kan = copy.deepcopy(model_kandy.model_).cpu()

# 2. Enable activation caching (PyKAN disables after training)
kan.save_act = True

# 3. Forward pass on subset to populate caches
n_sym = min(5000, len(data))
sym_input = torch.tensor(data[:n_sym], dtype=torch.float32)
kan(sym_input)

# 4. DO NOT call kan.prune() — breaks cache dimensions on small KANs

# 5. Run robust_auto_symbolic
robust_auto_symbolic(
    kan,
    lib=['x', 'x^2', 'x^3', '0'],
    r2_threshold=0.80,
    weight_simple=0.80,
    topk_edges=8,
    set_others_to_zero=True,
)

# 6. Extract and clean formula
formulas, _ = kan.symbolic_formula()
subs = {sp.Symbol(f'x_{i+1}'): sp.Symbol(n) for i, n in enumerate(FEAT_NAMES)}
raw = formulas[0].subs(subs)
expanded = sp.expand(raw)

# 7. Round: drop terms with |coeff| < tolerance, round to 3 places
COEFF_TOL = 0.01
terms = []
for term in sp.Add.make_args(expanded):
    coeff, rest = term.as_coeff_Mul()
    if abs(float(coeff)) > COEFF_TOL:
        terms.append(sp.Float(round(float(coeff), 3)) * rest)
cleaned = sum(terms)
```

**Why:** The squared-term artifacts (e.g. `-2.38*(4.02 - 0.0007*u_xx)² + 38.60`) arise because `fix_symbolic` fits `x^2` to near-constant edges. After expanding, the large constants cancel and only tiny corrections remain. The rounding step cleans this up.

**How to apply:** Use this procedure in any example script that needs symbolic equations. Apply the SAME `COEFF_TOL` to all methods (PDE-FIND, OLS, LASSO, KANDy) for fair comparison.
