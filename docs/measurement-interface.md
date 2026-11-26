# 测量接口：A 的候选 → B 的原始表 → C 的分析

## 1. A → B

B 加载 A 归档的生成代码，在计时前完成 Triton 初始化。接口参见
[`docs/interface-for-b.md`](interface-for-b.md)：

```python
from fusionboundary.runtime import CandidateStore
store = CandidateStore(runs/<candidate_run>)
cand  = store.load(case_id, "split" | "fused", device="cuda")
```

适配层 `fb/fusionboundary_provider.py` 使用：

| A 接口 | B 的用法 |
|---|---|
| `store.load(cid, strategy)` | 计时前一次加载；`load_ms`/`cache_reused` 记入 `variants.jsonl` |
| `cand.fixed_input()` | 两种策略共用同内容、地址稳定的输入 |
| `cand.call(x)` | 普通模式一次调用；批内 `call_n(n)` 连续发射，不插入同步 |
| `cand.capture_replay()` | replay 模式，一张图 = 一次完整调用 |
| `cand.verify()` | 4 组固定输入的独立数值复核，写入 `checks.jsonl` |
| `cand.report.kernel_count` | 校验 `variants.jsonl.kernel_count` 与发射数 |
| `variants.jsonl` | 状态、产物哈希、`default_maps_to`、边界证据的来源 |

B 侧强制检查（正式 run）：`artifact_sha256` 非空、`n_kernels` 为正、
输入地址稳定（`preallocated_inputs`）。每次 `load()` 后还会用一次非计时 profiler
观测实际发射的 kernel 名，并与 A 的 `kernels.json`/`kernel_names` 比对；
不一致直接拒绝计时（`prepare_error`），避免把 fused 代码当成 split 数据。
缺失契约字段时中止并提示 A 补产物；`--allow-unverified-candidates` 仅供
本地/合成试跑（reference provider 仍标记为未验证）。

B 在测量 run 中：

- 复制 A 的 `run.json`、`cases.jsonl`、`variants.jsonl`、环境快照到
  `artifacts/candidates/`；
- 将候选证据复制到 `artifacts/candidates/artifacts/<case>/<strategy>`。
  `run.json.candidates.source_run` 记录源 run，分析只需测量目录。
- 保留 A 的失败变体行；不能配对的 case 在 `cases.jsonl` 标
  `planned_status=excluded` + `exclusion_reason`。

若 A 的接口字段变化，A 会升 `__version__` 并在 `docs/CHANGELOG-interface.md`
记录；B 需要同步改适配层，不改 runner 的配对/重试/恢复逻辑。

## 2. B → C 的原始表

每个 run 一个目录 `runs/<run_id>/`，JSONL 一行一条记录。C 只应读取原始表，
不要依赖 B 的中间派生。

| 表 | 关键字段 | 用途 |
|---|---|---|
| `cases.jsonl` | `case_id, template, shape, dtype, t_shape, t_dtype, logical_saved_bytes, input_seeds, planned_status, exclusion_reason` | 语料与排除原因 |
| `variants.jsonl` | `strategy, status, artifact_path, artifact_hash, n_kernels, kernel_names, default_maps_to, t_materialized, edge_verdict` | 候选证据、x₁ 静态发射数、default 映射 |
| `checks.jsonl` | `case_id, strategy, input_id, mode, pass, max_abs_err, max_rel_err` | 正确性门槛；replay 至少一行 |
| `timings.jsonl` | `case_id, mode, block, attempt, strategy, iteration_order, n, elapsed_ms, us_per_call, quality, reason, gpu_uuid, artifact_hash` | 配对收益的原始单位 |
| `calibration.jsonl` | `mode, probe, k / buffer_bytes, block, repeats, elapsed_ms, us_per_call, derived` | L、L_replay、BW 原始行，可重算 |
| `telemetry.jsonl` | `ts_utc, temperature.gpu, power.draw, clocks..., clocks_throttle_reasons.active`, `kind=apps` 行 | 污染判定与机制 |
| `profiles.jsonl` | `case_id, strategy, kernel_index, metric, value, unit, source, trace_path` | 逐 kernel 诊断；缺测 `null` + reason |
| `verification.json` | `valid_pairs, failed_blocks, error_rows, contaminated_rows, problems` | 完整性/覆盖率对账 |

### 配对选择规则（与 03/04 一致）

1. 对 `(case_id, mode, block)` 取**最小的完整 attempt**：两条行、策略为
   split/fused、`quality=ok`、`n` 相同；
2. 30 个不同 block 都选中才算该 case/mode `valid`；
3. `T_split/T_fused` 为选中行的 `us_per_call` 均值；收益
   `B = mean(split_us - fused_us)`；
4. 重试 attempt、污染 attempt、缺一半的行全部保留，但不得进入主配对；
5. `mode=direct` 是主终点，`replay` 只作提交方式对照，二者不混合。

最小选择片段：

```sql
SELECT case_id, mode, block, n,
       max(CASE WHEN strategy='split' THEN us_per_call END) AS split_us,
       max(CASE WHEN strategy='fused' THEN us_per_call END) AS fused_us
FROM timings
WHERE quality = 'ok'
GROUP BY case_id, mode, block, n, attempt
HAVING count(*) = 2 AND count(DISTINCT strategy) = 2;
-- 再对每个 (case_id, mode, block) 取 min(attempt)
```

### 最小 synthetic 样例

`fixtures/measurement-synthetic/`：P1/N=65536/f32 与 R1/M=256/K=512/f32，
2 个 mode × 30 block × 2 策略 = 240 行配对 timings；另含 60 行**未配对**
`default`（P1 的 default 是第三种分区，`paired: false`），附 calibration、telemetry、
checks。
**每一行和 run.json 都标记 `synthetic: true`，数值由 FakeBackend 生成，不能进入报告。**
用途只是钉住字段、配对与校验查询。重新生成：

```bash
python -m fb synth-export --out fixtures/measurement-synthetic --force
```

独立校验任意 run：

```bash
python -m fb verify --run-dir runs/<run_id>
```
