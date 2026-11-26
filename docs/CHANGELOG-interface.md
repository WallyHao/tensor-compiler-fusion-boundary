# 接口变更记录（给 B）

接口以 `src/fusionboundary/runtime.py` 为准，`fusionboundary.__version__` 跟随语义变化。

## 0.1.0 — 2026-11-25

首次冻结：

* `CandidateStore(run_path)`：`load(case_id, strategy)`、`load_pair(case_id)`、
  `status()`、`variants()`、`case_ids()`。
* `Candidate`：`fixed_input()`、`inputs`、`call()`、`call_latest()`、`call_n()`、
  `fresh_inputs()`、`verify()`、`capture_replay()`、`report`、`expected_output`、
  `logical_edge_bytes`。
* `LoadReport`：`summary()` 返回纯 JSON；`cache_reused` 由隔离缓存目录的
  文件快照（load 前后）判定，`triggered` 列出 `compile_wrapper` / `first_call` /
  `cache_miss_kernels`，`tuning` 永不出现（配置固定在 `max_autotune* = False`）。
* 版本不符（torch / triton / CUDA）时 `load()` 直接报错，不静默重编译。
