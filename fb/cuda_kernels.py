"""Minimal Triton kernels used only by device calibration."""
from __future__ import annotations

try:
    import triton
    import triton.language as tl

    @triton.jit
    def noop_write(ptr):
        tl.store(ptr + tl.arange(0, 1), tl.zeros((1,), tl.int32))

    @triton.jit
    def copy_bytes(src, dst, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        val = tl.load(src + offs, mask=mask, other=0)
        tl.store(dst + offs, val, mask=mask)

except Exception:  # triton missing; calibration reports the error instead of crashing import
    triton = None
    noop_write = None
    copy_bytes = None
