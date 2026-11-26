# AOT ID: ['0_inference']
# SYNTHETIC FIXTURE — mirrors a persistent reduction with an in-place accumulator:
# one kernel takes the output buffer as `in_out_ptr0`, reads and writes it, and the
# wrapper aliases it (`buf1 = buf0`). A buffer with t's shape is therefore written
# and read by the *same* kernel; that must NOT be reported as t being materialised.
from ctypes import c_void_p, c_long, c_int
import torch
from torch._inductor.async_compile import AsyncCompile
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/ff/ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff.py
# Topologically Sorted Source Nodes: [t, relu], Original ATen: [aten.sum, aten.relu]
triton_per_fused_relu_sum_0 = async_compile.triton('triton_per_fused_relu_sum_0', '''
import triton
import triton.language as tl

@triton.jit
def triton_per_fused_relu_sum_0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0 + 512*x0), rmask & xmask, other=0.0)
    tmp1 = tl.sum(tmp0, 1)[:, None]
    tmp2 = tl.where(0*tmp1 < tmp1, tmp1, 0.0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (256, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [t, relu], Original ATen: [aten.sum, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_relu_sum_0.run(buf1, arg0_1, 256, 512, grid=grid(256), stream=stream0)
        del arg0_1
    return (buf1, )
