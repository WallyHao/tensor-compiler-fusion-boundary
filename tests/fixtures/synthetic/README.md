# SYNTHETIC FIXTURES — NOT EXPERIMENTAL DATA

These files mimic the shape of TorchInductor's generated artifacts (as of the
pinned torch version) so that `fusionboundary.forensics` can be unit tested
without a GPU and without running a compile. They are hand-written, are never
copied from a real run, and must not be used as evidence in the report.

Real artifacts (with `output_code_sha256`, GPU UUID and torch version) are
written by `python -m fusionboundary pilot` under `runs/<run_id>/artifacts/`.
