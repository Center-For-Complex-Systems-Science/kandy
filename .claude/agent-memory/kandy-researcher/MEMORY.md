# KANDy Researcher Agent Memory

**IMPORTANT:** Detailed experiment notes live in `notes/` at project root. Always check `notes/MEMORY.md` for the full index before running experiments.

## Memory Files
- [ieeg_lessons.md](ieeg_lessons.md) - Real iEEG data: v7 approximate vanishing ideal spline fitting (backward elimination, 7/8 edges R^2>0.76), v6 Mode 0 Duffing-ReLU (grid=3, lamb=0.001), run-to-run variability (6-8 active edges), auto_symbolic failure mode (updated 2026-03-12)
- [lift_design.md](lift_design.md) - Lift design patterns across different system types, including KAN sample-size requirements (updated 2026-03-12)
- [pdefind_comparison.md](pdefind_comparison.md) - PDE-FIND vs KANDy head-to-head on Burgers: KANDy wins (2 terms, NRMSE=0.049), PDE-FIND fails on shock data (all diff methods), derivative method/grid/IC settings that work (2026-03-17)
- [symbolic_extraction.md](symbolic_extraction.md) - Step-by-step procedure for robust_auto_symbolic: deepcopy, save_act=True, no prune, formula rounding with shared COEFF_TOL (2026-03-17)
