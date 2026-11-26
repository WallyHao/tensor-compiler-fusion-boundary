# 调用接口（给测量负责人 B）

本文件定义 B 侧唯一需要依赖的稳定接口。实现：`src/fusionboundary/runtime.py`。
语义、生命周期与"哪些步骤会编译或调优"都在这里写死；接口变化会升 `__version__`
并在 `docs/CHANGELOG-interface.md` 记录。

## 1. 两个对象

```python
from fusionboundary.runtime import CandidateStore

store = CandidateStore("runs/pilot-20261125")          # 读 variants.jsonl
cand  = store.load("R1-m256-k512-f32-contig-t", "split")
```

* `CandidateStore(run_path)`：只读索引，`variants.jsonl` 中 `status in {"valid","cache_only"}`
  的行才可加载。`store.status(case_id, strategy)`、`store.variants(case_id)`、
  `store.case_ids()`、`store.load_pair(case_id)`。
* `Candidate`：一个已编译候选。关键成员：

| 成员 | 语义 |
|---|---|
| `candidate.fixed_input(input_id="seed17")` | 设备上的固定输入，句柄持有，反复取回同一对象、同一地址 |
| `candidate.inputs` | `{"seed17","seed29","seed43","zeros"}` → 张量（同上，地址稳定） |
| `candidate.call(x=None)` | **一次**调用；`x=None` 时用 `seed17` |
| `candidate.call_latest(x=None)` | 一次调用，且只保留一个存活输出（地址稳定） |
| `candidate.call_n(n, x=None)` | 连续 n 次，逐次丢弃输出（无 host 同步） |
| `candidate.fresh_inputs(seed)` | 新分配输入；**只能放在计时区间外** |
| `candidate.report` | `LoadReport`，见 §3 |
| `candidate.verify()` | 4 组固定输入做数值校验，返回 `checks.jsonl` 行 |
| `candidate.capture_replay()` | 把"整次调用"捕获成一张 CUDA Graph，返回 `ReplayCandidate` |
| `candidate.expected_output` | `{"shape": [...], "dtype": "float32"}` |
| `candidate.logical_edge_bytes` | 逻辑节省字节（写一次 t + 读一次 t） |

`ReplayCandidate`：`.call()` = 一次 `graph.replay()`；`.call_n(n)`；`.verify()`
在 capture 后重放一次并与 eager 比较（`mode="replay"` 的记录行）。

## 2. 生命周期

```
load(case_id, strategy)                call(x) … call(x) …
├─ 重建同一个 torch.compile wrapper    ├─ Inductor 生成的 wrapper 内部
├─ 首次调用（唯一可能变慢的一步）      │   分配输出 buffer（empty_strided）
└─ 校验记录与 cache 快照              └─ 按记录顺序发射各 kernel
```

| 步骤 | 会不会编译 | 会不会调优 | 会不会分配 |
|---|---|---|---|
| `store.load()` | 核对哈希后加载已归档的 Inductor wrapper，Triton 可能生成设备二进制 | 固定配置关闭调优 | 预热并分配固定输入 |
| `call()` | 无 | 无 | 分配输出 |
| `fresh_inputs(seed)` | 无 | 无 | 分配输入 |
| `capture_replay()` | 无 | 无 | 创建 CUDA Graph 和静态输出 |

`report.first_call_ms` 包括加载、首次调用和同步。`cache_reused` 表示加载前后磁盘文件未变，
实际执行版本以归档 wrapper 的哈希和非计时 profiler 核验为准。
在普通启动和 CUDA Graph replay 中分别核对每次调用的 kernel 名称与发射次数。
PyTorch CUDA Graph 的 RNG offset 填充属于 replay 开销，单独记录。

## 3. `LoadReport`

```python
cand.report.summary()
# {'case_id': 'R1-m256-k512-f32-contig-t', 'strategy': 'split', 'status': 'valid',
#  'kernels': 2, 'kernel_names': ['triton_..._0', 'triton_..._1'],
#  'cache_reused': True, 'cache_added_files': 0, 'first_call_ms': 41.2,
#  'triggered': ['load_archived_wrapper', 'first_call'],
#  'evidence': '.../artifacts/R1-m256-k512-f32-contig-t/split/evidence.md',
#  'warnings': []}
```

`summary()` 是纯 JSON，可直接写进 `timings.jsonl` 的 `quality`/附加列。

## 4. B 必须遵守的不变量

1. **输入**：计时用 `candidate.fixed_input()`（或 `inputs[id]`）返回的对象，不要
   `clone()`、不要改内容、不要 `.cpu()`。两个策略必须使用**内容相同**的输入；同一
   `case_id` 的 `seed17` 与策略无关（输入在 CPU 生成、按 case dtype 转换后搬上设备）。
2. **输出**：`call()` 每次新分配输出。若需要固定地址（CUDA Graph capture、对齐
   显存流量），用 `call_latest()`（只保留一个存活输出）或 `capture_replay()`；普通
   计时按 03-collection 的"wrapper 正常工作"口径，用 `call()` 即可。
3. **地址稳定**：同一 `Candidate` 内 `fixed_input()` 地址不变；不同 `Candidate`
   （split / fused）地址不同，这是预期行为，配对比较的是"同样内容、同样 layout"。
4. **同步**：`call()` 不含同步（PyTorch 异步语义）。计时前自行
   `torch.cuda.synchronize()`，事件区间内不要插入同步。
5. **预热**按 03-collection 执行（≥50 次 + sync）；预热用 `call_n(50)` 即可，或
   自己循环 `call()` 并 `del` 输出。
6. **顺序**：同一 `case_id` 先 `load()` 再预热；不要在同一进程里交错 `load()` 与计时，
   `load()` 会做一次真实调用。

## 5. 状态与失败

* `store.status(case_id, strategy) is None` → 该组合没有可用产物：按 04-data 的口径
  记 `incomplete`/`unsupported`，不要自行编译。
* 失败原因在 `variants.jsonl` 的 `errors`、`status` 和对应
  `artifacts/<case_id>/<strategy>/compile_result.json`；切边证据在 `evidence.md`。
* `load()` 会核对 `compile_key.json` 里的 torch / triton / CUDA 版本；版本不一致直接
  报错而不是静默重编译。

## 6. 最小示例

```python
import torch
from fusionboundary.runtime import CandidateStore

store = CandidateStore("runs/pilot-20261125")
case_id, strategy = "P1-n65536-f32-contig-t", "split"
cand = store.load(case_id, strategy)
print(cand.report.summary())

x = cand.fixed_input()
cand.call_n(50)                      # 预热
torch.cuda.synchronize()

start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record()
cand.call_n(200)                     # 自己的重复数由 03-collection 的 5 ms 规则决定
end.record(); torch.cuda.synchronize()
print(start.elapsed_time(end))       # ms，除以 200 得 ms/call
```

同一 case 的两个策略：

```python
pair = store.load_pair(case_id)      # {"split": Cand, "fused": Cand}
xs = {s: c.fixed_input() for s, c in pair.items()}
```

试跑脚本：`examples/load_and_call.py`（只做 load + 一次调用 + verify，不包含任何计时协议）。

## 7. B 会读的表

`runs/<run_id>/variants.jsonl` 每行（`case_id, strategy`）：

| 字段 | 含义 |
|---|---|
| `status` | `valid` / `unsupported` / `partition_mismatch` / `incorrect` / `compile_error` |
| `kernel_count`, `kernel_names` | 生成代码中的 kernel 数与名字（发射顺序） |
| `t_materialized`, `t_alloc_dtype`, `t_alloc_shape` | split 证据：t 是否物化、物化类型/shape |
| `edge_verdict` | `materialized` / `fused` / `unclear` |
| `graph_count`, `graph_break_count` | 图断裂证据（控制手段是否引入额外 Python 调用） |
| `default_maps_to` | 默认策略实际等于哪个候选（`fused`/`split`/`distinct`） |
| `partition_equal_default_fused` 等 | 分区等价性，按生成代码签名比较 |
| `unrelated_partition_change` | 非目标分区是否被改动 |
| `checks_total`, `checks_failed` | 数值校验（4 组输入） |
| `artifact_dir` | 产物根，含 `output_code.py`、IR dump、`evidence.md` |

`checks.jsonl` 每行 `(case_id, strategy, input_id, mode)`，含 `max_abs_err`、
`max_rel_err`、`pass`、`fp64`（归约样本）。
