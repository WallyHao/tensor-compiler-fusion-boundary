# AOT ID: ['0_inference']
# SYNTHETIC FIXTURE — mirrors the pattern torch 2.6 emits for the split pilot:
# two kernels, the first allocates t and stores it, the second reads/rewrites the
# same storage through an in_out_ptr, and the wrapper hands the buffer over with
# `buf1 = buf0; del buf0  # reuse`. Never experimental evidence.
from ctypes import c_void_p, c_long, c_int
import torch
import math
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/dd/dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd.py
# Topologically Sorted Source Nodes: [mul, t], Original ATen: [aten.mul, aten.add]
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.25
    tmp2 = tmp0 * tmp1
    tmp3 = 0.25
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')

# kernel path: /tmp/torchinductor_root/ee/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
triton_poi_fused_relu_1 = async_compile.triton('triton_poi_fused_relu_1', '''
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.where(0*tmp0 < tmp0, tmp0, 0.0)
    tl.store(in_out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (65536, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((65536, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, t], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_0.run(arg0_1, buf0, 65536, grid=grid(65536), stream=stream0)
        del arg0_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf1, 65536, grid=grid(65536), stream=stream0)
    return (buf1, )


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', lambda times=10, repeat=10: None)
