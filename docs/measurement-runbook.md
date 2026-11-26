# 测量执行手册（B）· RTX 4090 配对采集

本文是 `fb` 测量程序的运行、验收与交接说明。协议口径来自
[`plan/phase1-4090/03-collection.md`](../plan/phase1-4090/03-collection.md) 与
[`04-data.md`](../plan/phase1-4090/04-data.md)。编译候选由 A 的
`fusionboundary` 包提供，接口见 [`docs/interface-for-b.md`](interface-for-b.md)，
本程序的适配层在 `fb/fusionboundary_provider.py`。

## 0. 不可违反的约束

- **GPU 独占**：校准、编译调优、正式计时、profiler、复跑不得重叠。所有 CUDA
  入口（`fb calibrate|collect|pilot|profile`）先取 `flock`，默认锁文件
  `<run_root>/gpu.lock`。A 的编译脚本不经过本锁，运行前必须人工确认没有 A 的
  `worker`/`compile` 进程；两边同时跑时后到者应在锁上等待，而不是并行。
- **设备**：正式运行要求 GPU 名称含 `RTX 4090`。名称或 UUID 不符时默认中止
  （`exit 2`），只有显式 `--allow-device-mismatch` 才继续，且 `run.json`
  会记录 `device_check.pass=false`。
- **持久存储**：`FB_RUN_ROOT` 必须指向实例释放后仍保留或可导出的卷。程序用
  `findmnt` 检查，位于 tmpfs/ramfs 或 `/tmp`、`/run` 下时正式运行中止
  （`exit 6`；仅供本地试跑时加 `--allow-ephemeral-root`）。
- **单位**：`elapsed_ms` 是 CUDA event 原始毫秒；`us_per_call =
  elapsed_ms * 1000 / n`；`bytes/µs = 1 MB/s`。所有派生值都能从原始行重算，
  `fb verify` 会重算并比对。
- **只追加**：每个 block attempt 先原子写 `blocks/*.json`，再追加
  `timings.jsonl`，再写 `state/*.json`。`run.json.protocol_hash` 不匹配时拒绝恢复。

## 1. 前置条件

```bash
cd ~/tensor-compiler-fusion-boundary
source .venv/bin/activate          # 或 PY=.venv/bin/python
export PYTHONPATH=$PWD/src          # 让 fb 的适配层能找到 fusionboundary
export FB_RUN_ROOT=/data/fusion-boundaries/runs   # 持久卷，实例释放后仍保留
# A 的候选 run（含 variants.jsonl、artifacts/<case>/<strategy>/）
export FB_CANDIDATE_RUN=$FB_RUN_ROOT/pilot-20261125T120000Z
# 可选：记录云平台信息
export FB_CLOUD_PLATFORM=... FB_INSTANCE_SPEC=... FB_IMAGE_ID=...
```

候选 run 必须由 A 用同一 `requirements.lock`（torch 2.6.0 / triton 3.2.0）编译。
`CandidateStore.load()` 会核对 `compile_key.json` 的版本，不一致会直接报错而不是
静默重编译；这是预期行为。

## 2. 环境记录与核对

```bash
python -m fb env --run-root "$FB_RUN_ROOT" --run-id env-$(date -u +%Y%m%dT%H%M%SZ) \
  --expect-device "RTX 4090"
```

写入 `environment/environment.json`、`environment/environment.lock.json`、
`environment/environment_raw/*`（`nvidia-smi -q`、`nvidia-smi -L`、
query-gpu/compute-apps、`lscpu`、`uname`、`df`、`findmnt`、`pip freeze`、
git diff）。必须核对：

- `gpu.name` / `gpu.uuid` / `torch.device.uuid`（两处 UUID 应一致）；
- `driver_version`、`torch.version.cuda`、`torch.version.git_version`、triton；
- `persistent_storage`（目标路径、fstype、是否 ephemeral）；
- `cuda_config` 实际取值：grad 关闭、cudnn benchmark 关闭、TF32 关闭、
  autocast 关闭、Inductor cudagraphs 关闭、单 stream；
- `environment.lock.json.freeze_sha256` 与 A 的 `requirements.lock` 一致。

设备名/UUID/镜像/持久路径任何一项与预期不符：不要采数据，先记录并修正实例。

## 3. 校准

正式采样内部默认已包含校准（`--calibrate`），也可单独跑：

```bash
python -m fb calibrate --run-root "$FB_RUN_ROOT" --run-id calib-... --expect-device "RTX 4090"
```

- **L, 普通启动**：同一 Triton launcher 连续发 k∈{1,8,32,128} 个最小 kernel，
  20 个独立 block，`repeats` 补到约 2 ms；每次 block 记录
  `elapsed_ms`、`repeats`、`calls`、`us_per_call`。对每个 k 的中位数做
  `t(k)=a·k+b` 拟合，`a` 单位为 µs/kernel，`b` 为每段固定开销。原始行在
  `calibration.jsonl`，拟合与残差在 `calibration_fit.json`。
- **L, replay**：对每个 k 手动 capture 含 k 次发射的图，记录 capture 毫秒
  （诊断列）与 replay 计时；同样 20 block。它与 direct 分开拟合。
- **BW, 有效复制带宽**：Triton copy kernel，S 从 256 B 倍增；
  `BW = 2S / 每次耗时`，单位 bytes/µs。每个尺寸固定缓冲区复用并核对输出一致。
  4090 的 L2 为 72 MiB，因此附带的尺寸扫描延伸到 512 MiB；主模型取
  **大于 L2 的最大三个尺寸**的中位数（`sizes_used_bytes` 会明确列出）。
  规范里“最大三个尺寸”的结果另存为 `spec_compliant_largest3_*` 供核对，
  小尺寸只作启动成本诊断。
- 验证：`launch_count_check` 用 profiler 统计 k=8 的发射数；`fit.valid=false`
  或斜率非正时，模型标为不可评估，不阻塞其余数据。

## 4. 试点（P1 + R1）

**一条可粘贴命令：**

```bash
FB_CANDIDATE_RUN=$FB_RUN_ROOT/pilot-20261125T120000Z bash scripts/measure_pilot.sh
```

脚本做两个阶段，证明“重启恢复不重复计入”：

1. `--stop-after-blocks 2` 提交两个配对 block 后正常退出（`status=paused`）；
2. 用同一 `RUN_ID` 再跑，跳过已完成键，补完剩余 block。

试点 = P1/N=65536/fp32 与 R1/M=256/K=512/fp32，两个提交模式，默认 5 个 block
（`BLOCKS=30` 可改成正式规模；正式 run 会重新测量试点，不沿用）。

试点检查清单：

- `verification.md` 中每个 case/mode 的 `valid/expected` 与 `SF/FS`；
- `timings.jsonl` 每对 split/fused 有相同 `n`，`us_per_call` 可由原始行重算；
- `variants.jsonl` 的 `kernel_count` 与 A 的 `kernels.json` 一致；
- 两种 mode 都有行，`replay` 的 `capture_ms` 在 `state/*.json`；
- 重启后 `attempt` 全为 1，没有重复 key；
- `calibration_fit.json` 的 direct/replay 斜率与带宽值合理；
- `telemetry.jsonl` 有 1 Hz 行；若外部进程/节流触发，相应 block 会整对重测。

试点通过后**单独归档**该 run（复制到 `$FB_RUN_ROOT/pilot-archive-...` 或导出），
正式采集必须使用新的 `run_id`。

## 5. 正式采集（可断开终端）

```bash
cd ~/tensor-compiler-fusion-boundary && mkdir -p "$FB_RUN_ROOT" && \
FB_CANDIDATE_RUN=$FB_RUN_ROOT/full-20261125T120000Z setsid nohup \
  bash scripts/measure_full.sh > "$FB_RUN_ROOT/full-launch.log" 2>&1 < /dev/null &
```

- `setsid nohup` 使作业脱离终端；断开网页终端/SSH 不影响。
- 用 `tail -f "$FB_RUN_ROOT/full-launch.log"` 观察；日志同时写入
  `$FB_RUN_ROOT/<run_id>.driver.log`。
- 崩溃/断点后恢复：`RUN_ID=<同一个 run_id> FB_CANDIDATE_RUN=... bash scripts/measure_full.sh`。
  协议哈希、seed、block 数一致才允许续跑。
- `run.json.status`：`done`（完整性检查通过）、`done_incomplete` 由
  `verification.md` 表达、`paused`（可用 `--stop-after-blocks` 主动暂停）。

正式协议（程序内固定并写入 `run.json.protocol`）：

| 项 | 口径 |
|---|---|
| 每 case/mode | 30 个配对 block，15 个 SF + 15 个 FS，种子固定后打乱 |
| n | 同一 case/mode 固定；0.5 s 稳定预热 → 32 次试算 → 10% 余量 → 至多两轮实批校正，使更快策略单批 ≥5 ms（`n_refine` 入 state） |
| 每批 | 前 5 次 untimed 调用 + sync；一对 CUDA events 包围整批；批后 sync 读 `elapsed_ms` |
| 预热 | 每 case/mode 每策略 ≥50 次 + sync（计时外） |
| 重试 | 报错/污染/缺一半 → 保留行，整对重测，最多 2 次补测；分析取首个完整合格 attempt |
| 遥测 | 约 1 Hz；block 窗口用最近的前后样本夹逼；外部 CUDA 进程或节流状态变化 → 污染 |
| 计时对象 | A 的已编译单图 `Candidate.call()`；replay 用 `capture_replay()` |
| 编译/加载 | 全部在计时外，`load_ms`/`cache_reused` 写入 `variants.jsonl` |

## 6. 机制诊断（profiler，单独进程）

```bash
python -m fb profile --run-dir "$FB_RUN_ROOT/$RUN_ID" --cases PROFILE --iters 10
```

`PROFILE` 预选 P1 最小/最大 N 与 R1 四个 (M,K) 角点（fp32）。产出
`profiles.jsonl`（每 kernel 的耗时/次数，缺测指标写 `null` + reason）、
Chrome trace 和 `artifacts/profiles/ncu-commands.sh`。若实例有 Nsight Compute
权限，按该脚本逐个跑；无权限时保留缺失原因即可，不影响主计时。profiler
必须在上一步全部结束后运行（脚本已串行）。

## 7. 数据与验收

```
$FB_RUN_ROOT/$RUN_ID/
  run.json  cases.jsonl  variants.jsonl  checks.jsonl
  calibration.jsonl  calibration_fit.json
  timings.jsonl  telemetry.jsonl  profiles.jsonl
  case_order.jsonl  state/  blocks/
  environment/  artifacts/candidates/  artifacts/profiles/
  verification.json  verification.md  freeze/manifest.json  SHA256SUMS
```

- `cases.jsonl` 覆盖 A 清单中的全部 case；不能成对的写入 `planned_status=excluded`
  与 `exclusion_reason`（缺失/无效变体）。
- `variants.jsonl` 保留 A 的失败变体行与证据链接（`artifacts/candidates/artifacts/...`）。
- `timings.jsonl` 一行 = 一个 block attempt 的一种策略；`quality` 取
  `ok|contaminated|error|pair_incomplete`。
- 失败清单：`verification.md` 的 `failed_blocks`、`error_rows`、
  `contaminated_rows`、`reasons`；原始失败行不会被删除。
- 校验和：`fb seal` 写 `SHA256SUMS` 与 `freeze/manifest.json`（含协议/语料/
  环境锁/校准的哈希）。导出后重新校验：

```bash
cd "$FB_RUN_ROOT/$RUN_ID" && sha256sum -c SHA256SUMS
```

## 8. 4090 云实例上的注意事项

1. 先确认实例规格与镜像是 4090 且已分配整卡；`nvidia-smi -L` 的 UUID 写入
   `run.json` 与每行 timing。
2. 功耗上限/时钟策略在采集期间不要改；遥测中的节流状态切换会把整块标污染。
3. 云监控进程可能占用 GPU；`nvidia-smi --query-compute-apps` 出现非本进程
   PID 时该 block 重测。
4. 4090 的 L2 为 72 MiB；不要用 64 MiB 以内的 copy 值当主 BW（会偏大），
   程序已按 L2 选择尺寸。
5. 持久卷路径：`FB_RUN_ROOT` 指向挂载点（如 `/data/...`），不要用 `$HOME`
   的 tmpfs 或容器 overlay。
6. 若 `torch.cuda.CUDAGraph` capture 失败，replay 行记
   `replay_capture_error`，direct 数据仍可进入主分析。
7. 实例/UUID 改变必须开新 run 并重新校准；不要跨 UUID 合并 timings。

## 9. 交付清单（本程序负责）

- 环境锁：`environment/environment.lock.json` + 原始查询 + `requirements.freeze.txt`；
- 运行命令：本文第 4、5 节两条一行命令；实际 `run.json.command_argv`；
- 校准原始行：`calibration.jsonl` + `calibration_fit.json`；
- 配对时间：`timings.jsonl`（含失败/重试 attempt）；
- 遥测：`telemetry.jsonl`；
- 失败清单：`verification.md` 的失败/排除段 + `verification.json`；
- 文件校验和：`SHA256SUMS` + `freeze/manifest.json`；
- 候选证据链接与 A 的清单副本：`artifacts/candidates/`；
- 最小 synthetic 配对样例（给 C）：`fixtures/measurement-synthetic/`，
  所有行 `synthetic: true`，**不是测量值**。

## 10. 本机（RTX 4060 Laptop）冒烟与已知阻塞

本机只用于验证流程，数据不进入报告。环境是 Nix，需要显式给出驱动库与编译器
（云端标准镜像不需要）：

```bash
export LD_LIBRARY_PATH=/nix/store/0cf8pz1kfhcnsc4017yjj410dhzkjd34-nvidia-x11-595.112.07/lib:$LD_LIBRARY_PATH
export TRITON_LIBCUDA_PATH=$LD_LIBRARY_PATH
export CC=/nix/store/ii6p6kc1brz3vycqg88q4zb4xd2dmqwp-gcc-wrapper-16.2.0/bin/gcc
export CXX=.../g++
PY=.venv-fb/bin/python bash scripts/measure_dryrun.sh   # reference 候选，3 block
```

2026-11-25 的实测阻塞（阻塞正式配对，不阻塞测量程序本身）：

- A 的 `split` 候选在 torch 2.6.0 下编译失败：
  `RuntimeError: torch._inductor.scheduler.SchedulerNode.can_fuse_vertical not found in this version`。
  该版本把 `can_fuse_vertical` 放在 `torch._inductor.scheduler.Scheduler` 上，
  `SchedulerNode` 没有这个属性。A 的 `_resolve_gate` 需要更新 gate 解析后重编；
  在修复前 `store.status(case, "split")` 为空，按接口约定 B 记 `excluded`，不自行编译。
- 本地 `default` 与 `fused` 已能编译并校验；测量层用 reference provider 完成了
  端到端 dry run（校准、两种 mode、遥测、落盘、恢复）。

## 11. 本机集成验证结果（2026-11-25，非 4090 数据）

| run | 候选 | 结果 | 用途 |
|---|---|---|---|
| `runs/env-local-3` | — | 设备/UUID/镜像字段齐全 | 环境记录 |
| `runs/dryrun-local-4` | reference | 12/12 配对（3 block） | 计时/模式/落盘 |
| `runs/pilot-local-ref` | reference | 20/20 配对；10 行污染 attempt；50 行唯一键 | 污染→整对重测、断点恢复 |
| `runs/pilot-adapter-local2` | A 的真实产物 | **已作废**：当时 split/fused 在进程内互相污染（见下） | 不计入任何结果 |

归档（不进报告）：`fixtures/pilot-4060-local/`（reference + 遥测/重试）。
A 候选的归档在 runtime 修复后重新采集，旧归档已删除。
本机校准（4060）：L_direct ≈ 9.6 µs/kernel，L_replay ≈ 0.85 µs/kernel，
BW ≈ 225 GB/s（取 64/128/256 MiB，L2=32 MiB）。4090 必须重测，不得沿用。

n 选择补充：0.5 s 稳定预热（仅 CUDA）→ 32 次试算 → 10% 余量 → 至多两轮
实批校正，保证较快策略单批 ≥5 ms；`n_refine` 与 `stabilize` 写入 state。
若 `default_maps_to="distinct"`，按 03 第 4 步用同一 n 补 30 条 **未配对**
default 行（`paired=false`），不参与配对。

## 12. 当前阻塞（2026-11-25）

1. **没有可用的 4090 实例信息**：仓库与环境里没有 IP/SSH/实例 UUID/镜像/持久卷
   挂载点，无法登录云端。`fb env --expect-device "RTX 4090"` 是登录后的第一条命令；
   设备名或 UUID 不符必须换实例。
2. **全量候选尚未编译**：A 的候选 run 需要 72 case × split/fused。当前只有两个
   试点在本地 4060 编译成功（split 2 kernel、fused 1 kernel，default=fused）。
   4090 上必须先由 A 用 `scripts/run_corpus.sh` 生成全量，再由 B 在同一持久卷上采。
3. **A 的 runtime 加载 split 时没有复现编译期的 edge realize/guard**（2026-11-25 23:54 复现）：
   实际发射的是 fused kernel。复现命令与验收条件见
   [`docs/measurement-blockers.md`](measurement-blockers.md)。
   `CandidateStore.load()` 先 split 再 fused 时，第二次 `torch.compile` 命中 Dynamo
   的进程内缓存，fused 实际执行 split 的两个 kernel（或反之）。B 已在适配层加
   发射校验：用 profiler 观测实际 kernel 名，与 `kernels.json` 不一致就拒绝计时。
   本地结论：split gate 已能编译，但 **A 必须修复 `CandidateStore.load`**，使
   split/fused 在同一个测量进程内各自执行自己的产物。B 的发射校验会拒绝计时，
   不会静默采错数据。

## 13. 启动前 GPU 独占检查

`fb collect|pilot` 启动时会查 `nvidia-smi --query-compute-apps`。只要存在非本进程
的 CUDA 进程就拒绝开始（`run.json.status=aborted_external_gpu`，`exit 8`），避免与
A 的编译/调优并行。正式 run 还禁止 `--allow-external-gpu` 与 `--no-telemetry`
（`exit 9`）；本地试跑用 `--purpose dryrun`。运行中才进入的进程由 1 Hz 遥测标记，
污染 attempt 会整对重测。A 的编译脚本不取 `gpu.lock`，所以仍需人工确认 A 已结束。
