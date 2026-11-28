# 融合资源悬崖：RTX 4090 上的 split 占优区（2026-11-28）

72-case 语料与 20 个补采用例的 direct 标签里没有 `split`，两图独立调度（`split_independent`）
也被否定（见仓库根 README 的「复核」）。本文件记录为回答“融合到底会不会亏”而设计的
stress family 及其结果：**在一条合法的单消费者边上，存在一个可复现的 split 占优区，
最深处 split 快 28.0 µs（相对 72.9%）。**

## 0. 发现摘要

1. **边界存在，但只在一个窄区。** 24 个配置中 10 个 `split`、13 个 `fuse`、1 个
   `uncertain`。split 占优区为 **M ∈ {512, 1024}、K = 8192、fp32、B ≥ 3**，最深
   `M=1024, B=8`：split 38.42 µs 对 fused 66.42 µs，**−28.0 µs（−72.9%）**。
   区间外融合仍领先：`M=4096, B=5` 时 +119.6 µs（fused 快 28.7%）。
2. **机制是 epilogue 的并行度被归约 tile 绑定。** fused 的 epilogue 在归约的
   `XBLOCK=1`、16 warps、grid=M 下运行，每个额外 load 约 +5 µs；split 的消费者是
   独立的点式 kernel（`XBLOCK=1024`、4 warps、grid=M·K/1024），每个 load 约 +0.9 µs。
3. **第二层损失来自 Inductor 的配置上限。** 当 `num_load + num_reduction ≥ 10` 时，
   `_reduction_configs` 置 `register_intensive`，fused 的 `RBLOCK` 从 2048 降到
   1024/512、warps 上限从 16 降到 8；split 的 producer 保持 2048/16。
4. **边界非单调。** M=256 由 launch 成本回到 fuse，M ≥ 2048 由行局部性/网格规模回到
   fuse；K=4096 未判定，K=2048 与 fp16 同一 (M,K) 回到 fuse。因此不能主张“融合普遍
   有害”，只能主张“在 epilogue 并行度被归约 tile 绑定的区域，融合会亏”。

## 1. 家族与切边

入口（单图、单输入、单输出）：

```
t = x.sum(dim=1, dtype=torch.float32)              # 行和，t 形状 (M,)
y = t[:, None] * (x + Σ_{s=1..B} roll(x, s, dims=1))   # 宽度 B 的 stencil epilogue
```

`B` 控制 epilogue 的独立 load 数：每个 `roll` 是一次不同的 `tl.load`，`B` 越大，fused
kernel 的 `num_load` 越大。`split` 是同一入口 + 自动生成的切边：lowering 阶段对 `t` 调
`realize()`，调度器否决该边再融合（`scripts/probe_cliff.py --role split`）。两侧输出
逐次 `allclose`（`outputs_match=true`）。

这个家族针对的是 Inductor 2.6 的两个具体行为（源码位置见下）：

1. 融合决策默认不看寄存器/占用率（`benchmark_fusion=False`），而 launch config 在融合
   **之后**按 `num_load + num_reduction` 选择。消费者每多一个独立 load，就把 fused
   kernel 推向 `register_intensive` 分支。
2. fused kernel 的 epilogue 继承归约的 launch config（`XBLOCK=1`、16 warps、grid=M），
   而 split 的消费者是一个独立的点式 kernel（`XBLOCK=1024`、4 warps、grid=M·K/1024）。

## 2. 采集协议

- 设备：`NVIDIA GeForce RTX 4090`（128 SM、72 MiB L2、24 GB），`torch 2.6.0+cu124`、
  `triton 3.2.0`、Python 3.12.3。
- 编译：fused 与 split 各在独立进程、独立 cache/trace 目录中编译（与
  `strategies.compile_variant` 相同的 pinned Inductor config）。
- 计时：同一进程加载两侧归档 wrapper，30（`B=3` 为 60）个配对 block，正反顺序交替，
  每 block 10 次调用；区间为 10,000 次 block bootstrap 的 95% 分位。
- 逐 kernel 资源来自编译产物（`launchers[0]`：block/warps/regs/spills/occupancy），
  逐 kernel 时间来自 `torch.profiler` 的 CUDA self time（`--profile`）。
- 标签沿用论文口径：相对收益 CI 整体高于 +2% 为 `fuse`，整体低于 −2% 为 `split`，
  整体落在 ±2% 为 `equivalent`，其余为 `uncertain`。

## 3. 结果

`runs/cliff-20261128T090000Z/`（`summary.json` 为机器可读表）。`B` 为 stencil 宽度；
`B=0` 即 `y = t[:,None] * x`。

| M | K | dtype | B | T_fused | T_split | B (split−fused) | rel | 95% CI (rel) | label |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 256 | 8192 | f32 | 5 | 16.96 | 28.34 | +11.38 | +40.2% | [+39.0, +41.4] | fuse |
| 512 | 8192 | f32 | 5 | 28.92 | 27.43 | −1.49 | −5.4% | [−6.9, −3.5] | split |
| 512 | 8192 | f32 | 8 | 31.87 | 26.22 | −5.65 | −21.5% | [−22.5, −20.3] | split |
| 1024 | 2048 | f32 | 5 | 16.19 | 28.40 | +12.21 | +43.0% | [+41.9, +44.2] | fuse |
| 1024 | 4096 | f32 | 5 | 27.56 | 28.67 | +1.11 | +3.9% | [−3.4, +12.3] | uncertain |
| 1024 | 8192 | f32 | 0 | 21.88 | 28.53 | +6.65 | +23.3% | [+23.1, +23.5] | fuse |
| 1024 | 8192 | f32 | 1 | 27.35 | 30.08 | +2.73 | +9.1% | [+8.4, +10.2] | fuse |
| 1024 | 8192 | f32 | 2 | 27.41 | 29.30 | +1.89 | +6.4% | [+6.1, +6.8] | fuse |
| 1024 | 8192 | f32 | 3 | 38.82 | 30.13 | −8.70 | −28.9% | [−29.1, −28.6] | split |
| 1024 | 8192 | f32 | 4 | 39.02 | 29.87 | −9.15 | −30.7% | [−31.0, −30.2] | split |
| 1024 | 8192 | f32 | 5 | 51.34 | 30.92 | −20.42 | −66.1% | [−66.4, −65.7] | split |
| 1024 | 8192 | f32 | 6 | 49.63 | 32.87 | −16.76 | −51.0% | [−52.2, −48.9] | split |
| 1024 | 8192 | f32 | 7 | 56.23 | 37.60 | −18.62 | −49.5% | [−49.8, −49.2] | split |
| 1024 | 8192 | f32 | 8 | 66.42 | 38.42 | −28.00 | −72.9% | [−73.2, −72.5] | split |
| 1024 | 8192 | f32 | 10 | 69.72 | 46.46 | −23.26 | −50.1% | [−50.6, −49.5] | split |
| 1024 | 8192 | f32 | 12 | 78.54 | 50.22 | −28.32 | −56.4% | [−56.7, −56.1] | split |
| 1024 | 8192 | f16 | 5 | 24.25 | 31.02 | +6.77 | +21.8% | [+17.9, +26.6] | fuse |
| 1024 | 32768 | f16 | 5 | 248.74 | 278.98 | +30.24 | +10.8% | [+10.7, +10.9] | fuse |
| 2048 | 8192 | f32 | 5 | 155.76 | 204.46 | +48.70 | +23.8% | [+23.3, +24.4] | fuse |
| 2048 | 8192 | f32 | 8 | 162.89 | 200.05 | +37.16 | +18.6% | [+18.4, +18.7] | fuse |
| 2048 | 8192 | f32 | 12 | 163.95 | 200.60 | +36.65 | +18.3% | [+17.4, +19.2] | fuse |
| 4096 | 8192 | f32 | 5 | 297.11 | 416.73 | +119.62 | +28.7% | [+28.6, +28.8] | fuse |
| 4096 | 8192 | f32 | 8 | 318.68 | 415.67 | +96.99 | +23.3% | [+23.2, +23.4] | fuse |
| 4096 | 8192 | f32 | 12 | 321.57 | 416.79 | +95.23 | +22.9% | [+22.7, +23.1] | fuse |

24 个配置中 10 个 `split`、13 个 `fuse`、1 个 `uncertain`。split 占优区是
**M ∈ {512, 1024}、K = 8192、fp32、B ≥ 3**；它是非单调的（M=256 与 M≥2048 都回到
fuse），也不是简单的字节阈值（K=4096 uncertain，K=2048 fuse，f16 同一 (M,K) 回到 fuse）。

## 4. 机制

**逐 kernel 时间（M=1024, K=8192, f32，µs/次调用，`torch.profiler` CUDA self time）**

| B | fused kernel | split producer | split consumer | fused 合计 | split 合计 |
|---:|---:|---:|---:|---:|---:|
| 0 | 20.96 | 8.77 | 18.37 | 21.88 | 28.53 |
| 3 | 37.15 | 9.02 | 19.09 | 38.82 | 30.13 |
| 5 | 50.26 | 8.76 | 20.55 | 51.34 | 30.92 |
| 8 | 65.18 | 8.88 | 26.97 | 66.42 | 38.42 |

- split 的 producer 恒定在 ~9 µs；它的消费者随 B 从 18.4 涨到 27.0 µs（每多一个
  load 约 +0.9 µs）。
- fused kernel 从 21.0 涨到 65.2 µs（每多一个 load 约 +5 µs）。同样的 epilogue，在
  fused 里比在 split 的独立点式 kernel 里贵约 5 倍。
- 原因：fused kernel 的 grid 是 `M`（每行一个 program），epilogue 继承归约的
  `XBLOCK=1`、16 warps；split 的消费者是 `XBLOCK=1024`、4 warps、grid 为
  `M·K/1024`。epilogue 的并行度被绑在归约输出数 `M` 上，而不是元素数 `M·K`。

**launch config 的第二层损失（`B ≥ 8`）**

fused kernel 的 `num_load + num_reduction` 随 B 增长；当 `M ≥ 1024` 且该和 ≥ 10 时，
`runtime/triton_heuristics.py:1657-1699` 的 `_reduction_configs` 置
`register_intensive=True`，把 `MAX_RBLOCK` 从 2048 降到 1024、`num_warps` 上限从 16
降到 8；`dynamic_scale_rblock` 再追加一个 `RBLOCK//2` 候选（`B=10/12` 时选中 512）。
split 的 producer 只有 `num_load=1 + num_reduction=1`，始终保留 `RBLOCK=2048, 16 warps`。
实测：

| B | fused config | fused regs/occ | split producer |
|---:|---|---|---|
| 5 | `RBLOCK=2048, 16w` | 40 / 1.0 | `RBLOCK=2048, 16w`, 26 regs / 1.0 |
| 6 | `RBLOCK=2048, 16w` | 55 / 0.67 | 同上 |
| 8 | `RBLOCK=1024, 8w` | 40 / 1.0 | 同上 |
| 10 | `RBLOCK=512, 8w` | 40 / 1.0 | 同上 |

占用率下降（B=6）与 config 上限（B≥8）是同一根因的两个出口：融合把消费者并进归约
kernel，归约的 tile 不再独立于 epilogue。

## 5. 为什么不是所有形状都翻转

- **M=256**：epilogue 总量小，两次 launch + t 物化的固定成本占主导，fuse 快 11.4 µs。
- **M≥2048**：fused 的 grid 足够大（每 SM 16–32 个 program），epilogue 的行局部性
  在同一个 kernel 内保留；split 的消费者要重新走一遍 `x`，融合反而快 37–120 µs。
- **K=2048**：行太短，epilogue 工作量小；**K=4096** 的 CI 跨 0，未判定。
- **fp16**：同一 (M,K) 下字节减半、归约的 tile 不同，fused 仍然快 6.8 µs。

## 6. 局限

- 单设备（RTX 4090）、单一家族；没有 H800/3090 的对照（3090 已不可用，归档里也没有该家族）。
- 没有硬件计数器：occupancy 来自 `cuOccupancyMaxActiveBlocksPerMultiprocessor`，时间来自
  `torch.profiler`；DRAM/L2 流量未测。
- 占优区窄且非单调；把 `B` 或 `M` 移动一格就可能回到 fuse。这本身是结论的一部分：
  边界不是单调阈值，而是“epilogue 并行度 × 归约 tile × 工作集”的交互。
- `x3` 静态代理（占用率差）在 B=6 与 B≥8 与符号一致，但在 M=4096、B=8（fused 被 cap、
  occupancy 0.83）时符号相反；静态代理不能单独解释符号。

## 7. 复现

```bash
PYTHON=/path/to/venv/bin/python bash scripts/run_cliff.sh
# 结果：runs/cliff-<UTC>/summary.json 与逐配置 json/
```

单配置：

```bash
PYTHONPATH=src python scripts/probe_cliff.py --variant wide \
  --M 1024 --K 8192 --dtype float32 --branches 5 --role fused --out /tmp/cliff
PYTHONPATH=src python scripts/probe_cliff.py --variant wide \
  --M 1024 --K 8192 --dtype float32 --branches 5 --role split --out /tmp/cliff
PYTHONPATH=src python scripts/probe_cliff_paired.py \
  --fused-code /tmp/cliff/wide-M1024-K8192-float32-b5__fused_output_code.py \
  --split-code /tmp/cliff/wide-M1024-K8192-float32-b5__split_output_code.py \
  --M 1024 --K 8192 --dtype float32 --branches 5 --out /tmp/cliff --profile
```

## 8. 对论文的含义

- 现在有 10 个 `split` 标签（24 个配置），标签表不再退化；可以用既有两项/三项模型在
  该家族上比较，检验 `x3`（资源代理）是否只在占优区有解释力。
- 不能把结论推成“融合普遍有害”：同一家族 13/24 是 fuse，M=4096 时融合领先 22.9–28.7%。
- 论文可以主张：融合的收益边界由“epilogue 的并行度是否被归约 tile 绑定”决定；在
  本家族上该边界可测、可复现，且与 Inductor 的 `register_intensive` 配置上限叠加。

## 9. 数据清单

| 文件 | 内容 |
|---|---|
| `data/fb-cliff-4090-20261128.tar.zst` | 原始数据：24 个配置的编译证据（`output_code.py`、逐 kernel 资源、`num_load`/`num_reduction`）、30/60 配对 block 的 A/B JSON、逐 kernel profiler 时间、`summary.json`；排除 `*/cache/` |
| `data/cliff-4090-results.csv` | 24 行扁平表：两侧时间、$B$、相对收益与 CI、标签、fused/split 的 block/warps/regs/occupancy、`num_load`、guard/realize 状态 |
| `runs/cliff-20261128T090000Z/summary.json` | 上述表的机器可读版本（解包后） |
| `data/fb-cliff-4090-20261128.tar.zst.sha256` | 归档校验和 |
