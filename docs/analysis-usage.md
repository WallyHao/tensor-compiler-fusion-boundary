# 分析入口与数据审计

实现位于 `fusionaudit/`，只依赖 Python 3.12 标准库。原始数据保持在 `runs/<run_id>/`，派生表、图和报告写入独立目录。

## 一条命令重建全部表图

```bash
.venv/bin/python -m fusionaudit build \
  --run runs/<run_id> \
  --out analysis/out/<run_id>
```

正式 3090 run 使用同一条命令；`--strict` 在审计出现 error 时返回退出码 2。`--expected-blocks 30 --bootstrap 10000 --seed 2026` 为默认口径。禁止把 `--out` 指向原始 run 目录。

已生成（2026-11-26，RTX 3090）：对 `runs/full-meas-3090-20261126T041621Z`
执行 `build --strict` 得到 **144 个 case/mode 结果、0 error / 0 warning**，输出在
`analysis/out/full-meas-3090-20261126T041621Z/`（`report.md` + 10 张 SVG）。direct 标签
68 `fuse` / 3 `equivalent` / 1 `uncertain` / 0 `split`；`always_fused`、默认 Inductor 与
两/三项模型 regret 均为 0 µs，`always_split` 平均 10.97 µs。

产物：

| 路径 | 内容 |
|---|---|
| `audit.json` / `audit_issues.csv` | 行级审计、外键/哈希/单位/配对/覆盖问题 |
| `block_audit.csv` | 逐 case/mode/block/attempt 的选择结果、原始行号、拒绝原因 |
| `results.jsonl` / `results.csv` | 每个完整 case/mode 的 T_split、T_fused、B、相对收益、95% 区间、四类标签、敏感性 |
| `features.jsonl` / `features.csv` | 白名单预测特征 x1/x2/x3 及来源 |
| `decisions.jsonl` / `decisions.csv` | 每折、每方法、每 case 的预测、覆盖率依据和 regret |
| `models.json` | 四折模型、全数据重拟合、字节阈值、系数/秩/条件数 |
| `provenance.json` | 逐 case/mode 的 block 行号、elapsed/n、artifact 路径与哈希 |
| `report.md` | 单卡报告；含覆盖率、审计、收益、决策评估、图和追溯实例 |
| `figures/*.svg` | 逐元素 N 曲线；归约 K×M 热图；覆盖、regret、direct/replay 对照 |
| `freeze/` | 冻结清单、特征白名单、模型/阈值、校验和、复跑说明 |

## 审计与选择规则

- 必需表：`run.json`、`cases.jsonl`、`variants.jsonl`、`checks.jsonl`、`timings.jsonl`、`calibration.jsonl`。`results.jsonl`、`telemetry.jsonl`、`profiles.jsonl` 可缺。
- `us_per_call` 不信任存储值：由 `elapsed_ms * 1000 / n` 重算；不一致记为 error。
- 一个 block/attempt 必须同时有 split 与 fused、两边 `n` 相同且为正、`artifact_hash` 与对应 variant 一致、对应策略在对应 mode 的 correctness check 全部通过、quality 合格。
- 同一 block 的 attempt 按编号排序，采用**第一个完整且合格**的 attempt；之前失败的整对丢弃，之后的 attempt 标记 `not_used`，绝不跨 attempt 拼接半对。
- 每 case/mode 必须有 `expected_blocks`（默认 30）个不同 block，block id 覆盖 `0..expected_blocks-1`；否则记 `incomplete_case`，不进入主结果。
- 检查顺序平衡（SF/FS 各约一半）与时间窗；异常保留为 warning。
- 验证 artifact、trace、环境锁、GPU 快照存在且哈希一致；失败行也保留。
- 输入 `results.jsonl` 若带 `fold`、`train_case_ids`、`test_case_ids`，会检查同模板是否跨训练/测试；发现 `template_leak` 即拒绝该指定。主评估始终按 `template` 重建 leave-one-template-out，不复用输入 fold。

## 合成夹具与四个情形

```bash
.venv/bin/python -m fusionaudit fixtures --out runs --force
.venv/bin/python -m fusionaudit selftest
.venv/bin/python -m fusionaudit build --run runs/synthetic-fixture-20261125 --out analysis/out/synthetic-fixture-20261125
```

夹具在 `run.json` 与所有报告中明确标记 `synthetic=true`，数值不可引用。`selftest` 断言：

1. `P1-n65536-f16-contig-t` 的 block 2 缺 fused 半边：记录 `pair_missing_half`，case 不进入结果。
2. `P2-n16384-f32-contig-t` 的 block 0/1 在 attempt 1 失败或污染，attempt 2 整对重测：选中 `0:2,1:2,2:1,3:1`。
3. `R1-m64-k512-f32-contig-t` 的相对收益 95% 区间落在 ±2% 内：标签为 `equivalent`；同时验证 `fuse`、`split`。
4. 输入 fold 表把两个 R2 case 放在训练/测试两侧：记录 `template_leak`；重建的四折训练/测试模板不相交。
5. 同一夹具自动改写为采集端拼写（list shape、`logical_save_bytes`、`seeds`、`artifact_dir`、`output_code_sha256`、`n_kernels`、`default_maps_to`、`iteration_order` 列表、`window_start_utc`、`kind=launch/bandwidth`），再构建一次并断言无 `invalid_check`/`invalid_timing`/哈希错误，shape 与校准则能正常归一化。

## 标签、模型与指标

- `B = mean(s_i - f_i)`；相对收益 `B / T_split`；区间为 30 个整对 block 的 10,000 次有放回 bootstrap（seed 2026）。
- ±2% 为预设带：区间整体高于 +2% 为 `fuse`，整体低于 −2% 为 `split`，整体落在带内为 `equivalent`，跨带为 `uncertain`；另给 ±1%、±5%。
- 留出：每次留出一个完整模板。默认策略、always fused、always split、字节阈值 τ、两项模型、三项模型在同一折比较；同时报覆盖率、符号准确率和 `regret = T_chosen − min(T_split,T_fused)` µs/call，并单列默认策略可映射的同一组 ID。
- 模型只用 `x1_us`、`x2_us`、`x3_us`。fused 实测耗时、实测差、运行后 occupancy、实测 DRAM/L2 流量、由实测选出的最佳 kernel 均不得进入特征；`models.json`/`freeze/features.json` 保存白名单和禁用字段。

## 追溯

```bash
.venv/bin/python -m fusionaudit trace \
  --run runs/synthetic-fixture-20261125 \
  --case-id R1-m256-k2048-f32-contig-t \
  --mode direct
```

输出包括每个 block 的 attempt、`timings.jsonl` 行号、`elapsed_ms`、`n`、重算 µs、split/fused artifact 路径与 SHA-256、correctness check 行号。报告第 8 节自动选择一条结果并展示同样的追溯表。

## 正式 run 的交接字段

`run.json` 建议显式写入：

```json
{
  "expected_blocks": 30,
  "expected_case_count": 72,
  "planned_template_counts": {"P1": 12, "P2": 12, "R1": 24, "R2": 24},
  "calibration": {
    "launch_us_per_kernel": {"direct": 4.0, "replay": 0.8},
    "bandwidth_bytes_per_us": {"direct": 500.0, "replay": 500.0},
    "l2_bytes": 100663296
  }
}
```

未写的校准系数会从 `calibration.jsonl` 推导：launch 使用各 k 的中位数做带截距线性拟合取斜率；BW 取最大三个尺寸的 `2*S*repeats/(elapsed_ms*1000)` 中位数。若存在 `calibration_fit.json` 或 `run.calibration.summary`，优先采用其 `launch_*.slope` 与 `bw_main_bytes_per_us`。`status=formal/main/production` 或 purpose 含 full/corpus/formal 时，缺少期望计数会按冻结语料默认 72/12/12/24/24 检查并给 error。

采集端的现有拼写可直接读取，不需要改原始文件：

| 数据契约 | 兼容别名 |
|---|---|
| `started_at` / `ended_at` | `started_utc` / `finished_utc` |
| `command` | `command_argv`（列表自动拼接） |
| `code_hash` | `run.code` 的 JSON 哈希 |
| `environment_lock` / `gpu_snapshot` | `run.environment.path`、`environment/environment.lock.json`、`environment/environment.json` |
| `input_seeds` | `seeds` |
| `logical_saved_bytes` | `logical_save_bytes` |
| `shape` `{N}` / `{M,K}` | `[N]` / `[M,K]`，并优先用 `n`/`m`/`k`/`grid_axes` |
| `artifact_hash` | `artifact_sha256`、`output_code_sha256` |
| `artifact_path` | `artifact_dir`（目录）、`output_code`（文件） |
| `static_launch_count` | `n_kernels`、`kernel_count` |
| `kernels` | `kernel_list` |
| `verified_default_decision` | `default_maps_to`、`default_equivalent`、`partition_equal_default_*` |
| `window.start/end` | `window_start_utc` / `window_end_utc` |
| `iteration_order` `SF`/`FS` | `["split","fused"]` / `["fused","split"]`，或 `block_pattern` |
| `quality=valid` | `quality=ok` |
| calibration `probe=launch` / `copy` | `kind=launch` / `kind=bandwidth`、`probe=L` / `probe=BW` |

`checks.jsonl` 中的 `default`、`mode=load` 行会被接受但不进入 split/fused 配对；独立 `default` timing 同样只作描述。


## 冻结与跨设备

`freeze/manifest.json` 固定输入/输出/分析代码哈希、特征白名单、分组、折、阈值和模型；`freeze/sha256sums.txt` 可校验整个派生目录。跨设备验证未执行，仅在 `freeze/README.md` 列出流程：重新校准 L/BW、重算 x1/x3、先写带时间戳的 split-only 预测、再解封 fused 标签。
