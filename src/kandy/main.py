#!/usr/bin/env python
"""kandy.main — CLI entry points for running KANDy experiments.

Agent orchestration (running experiments, reviewing results, generating
reports) is handled by Claude Code agents, not by an in-process framework.
These entry points are thin wrappers for common operations.

Usage:
    uv run kandy <system>           # run a KANDy experiment
    uv run kandy --list             # list available systems
    uv run kandy-baselines <system> # run baselines for a system
"""
import sys
import importlib
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_RESEARCH_DIR = Path(__file__).resolve().parents[2] / "research_code"

# Map system names to example scripts
_SYSTEMS = {
    "lorenz":               "lorenz_example",
    "henon":                "henon_example",
    "burgers":              "burgers_example",
    "burgers-fourier":      "burgers_fourier_example",
    "kuramoto-sivashinsky": "kuramoto_sivashinsky_example",
    "hopf":                 "hopf_example",
    "ikeda":                "ikeda_example",
}

_BASELINES = {
    "sindy":           "sindy_baselines",
    "pdefind":         "pdefind_baseline",
    "burgers-fourier": "burgers_fourier_baselines",
}


def _run_one(system: str):
    """Run a single system by name."""
    import runpy
    import os

    module_name = _SYSTEMS[system]
    script = _EXAMPLES_DIR / f"{module_name}.py"
    if not script.exists():
        print(f"Script not found: {script}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Running: {system}")
    print(f"{'='*60}")
    os.chdir(script.parent.parent)
    runpy.run_path(str(script), run_name="__main__")


def run():
    """Run a KANDy experiment by system name."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-l"):
        print("Available systems:")
        for name in sorted(_SYSTEMS):
            print(f"  {name}")
        print("\nUse --all to run every system.")
        return

    if sys.argv[1] in ("--all", "-a"):
        for system in _SYSTEMS:
            _run_one(system)
        print(f"\n{'='*60}")
        print(f"  All {len(_SYSTEMS)} systems complete.")
        print(f"{'='*60}")
        return

    system = sys.argv[1].lower()
    if system not in _SYSTEMS:
        print(f"Unknown system '{system}'. Use --list to see options.")
        sys.exit(1)

    _run_one(system)


def run_baselines():
    """Run baseline comparison scripts."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-l"):
        print("Available baselines:")
        for name in sorted(_BASELINES):
            print(f"  {name}")
        return

    name = sys.argv[1].lower()
    if name not in _BASELINES:
        print(f"Unknown baseline '{name}'. Use --list to see options.")
        sys.exit(1)

    module_name = _BASELINES[name]
    # Check examples/ first, then research_code/
    for d in (_EXAMPLES_DIR, _RESEARCH_DIR):
        script = d / f"{module_name}.py"
        if script.exists():
            break
    else:
        print(f"Baseline script not found for '{name}'")
        sys.exit(1)

    print(f"Running: {script}")
    import runpy
    import os
    os.chdir(script.parent.parent)
    runpy.run_path(str(script), run_name="__main__")
