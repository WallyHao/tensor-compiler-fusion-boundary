# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride

async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.py
# Source Nodes: [add, mul, relu], Original ATen: [aten.add, aten.relu]
triton_poi_fused_add_mul_relu_0 = async_compile.triton('triton_poi_fused_add_mul_relu_0', '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_mul_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 1.25
    tmp2 = tmp0 * tmp1
    tmp3 = 0.25
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(0*tmp4 < tmp4, tmp4, 0.0)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (65536, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [add, mul, relu], Original ATen: [aten.add, aten.relu]
        buf0 = empty_strided((65536, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, mul, relu], Original ATen: [aten.add, aten.relu]
        triton_poi_fused_add_mul_relu_0.run(arg0_1, buf0, 65536, grid=grid(65536), stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((65536, ), (1, ), device='cuda', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
