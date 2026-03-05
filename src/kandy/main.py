#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from kandy.crew import Kandy

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Supported systems (maps to research_code/ experiments):
#   'Lorenz'               — Lorenz (3).ipynb
#   'Henon'                — henon.py
#   'Burgers'              — Inviscid_Burgers (1).ipynb
#   'Burgers-Fourier'      — Inviscd-Burgers-fourier-mode-ics.ipynb
#   'Kuramoto-Sivashinsky' — Kuramoto–Sivashinsky (1).ipynb
#   'Navier-Stokes'        — Navier-Stokes.ipynb
#   'Hopf'                 — hopf.ipynb

INPUTS = {
    'system': 'Lorenz',
    'current_year': str(datetime.now().year),
}


def run():
    """Run the full KANDy research crew (clean → generate → package → review)."""
    try:
        Kandy().crew().kickoff(inputs=INPUTS)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    try:
        Kandy().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=INPUTS,
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        Kandy().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and return results."""
    try:
        Kandy().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=INPUTS,
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """Run the crew with a JSON trigger payload (for external orchestration)."""
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument.")

    inputs = {
        **INPUTS,
        "crewai_trigger_payload": trigger_payload,
        # Override system from payload if provided
        "system": trigger_payload.get("system", INPUTS["system"]),
    }

    try:
        return Kandy().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
