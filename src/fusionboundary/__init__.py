"""Fusion Boundaries phase-1 tooling.

Modules
-------
corpus     72-case manifest (templates x shape grid x dtype), edge metadata, logical bytes.
templates  eager reference expressions and the single-graph entry points used by all strategies.
strategies default / split / fused TorchInductor compilation of the same graph.
forensics  parsing of Inductor trace output: kernels, grid/resources, t write/read, partition diff.
validate   fixed-seed inputs, eager + fp64 references, tolerance table, error records.
runtime    stable loader/call interface consumed by the measurement owner (Agent B).
records    run directory layout, JSONL writers, hashes.
cli        `python -m fusionboundary ...` entry points.
"""

__version__ = "0.1.0"
