---
name: list-systems
description: List all available KANDy systems, baselines, and show current experiment results status.
---

# List Available Systems & Results

## Available KANDy Systems

!`uv run kandy --list 2>/dev/null`

## Available Baselines

!`uv run kandy-baselines --list 2>/dev/null`

## Current Results

!`for d in results/*/; do echo "## $(basename $d)"; ls "$d" 2>/dev/null | head -10; echo ""; done 2>/dev/null || echo "No results yet"`

## Example Scripts

!`ls examples/*.py 2>/dev/null`

## Experiment Notes

!`ls notes/*.md 2>/dev/null || echo "No notes yet"`

Summarize what's available and what results exist so far. Highlight any systems that haven't been run yet.
