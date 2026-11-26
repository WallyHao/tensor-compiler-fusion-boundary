# 72-case manifest（生成物，勿手改）

`phase1-72-cases.jsonl` 由代码生成：

```bash
PYTHONPATH=src .venv/bin/python -m fusionboundary manifest --run runs/manifest-<UTC>
cp runs/manifest-<UTC>/cases.jsonl manifest/phase1-72-cases.jsonl
```

`tests/test_manifest_sync.py` 会重新生成并逐字节比较；清单与 `src/fusionboundary/corpus.py`
不一致时测试失败。

布局（`sha256` 见上表命令输出，也在 `runs/manifest-<UTC>/cases.json`）：

| 模板 | 数量 | 形状网格 |
|---|---|---|
| P1 | 12 | N ∈ {1024, 8192, 65536, 524288, 4194304, 16777216} × {f32, f16} |
| P2 | 12 | 同上 |
| R1 | 24 | M ∈ {16, 256, 4096} × K ∈ {128, 512, 2048, 8192} × {f32, f16} |
| R2 | 24 | 同上 |

每行字段：`case_id`、`template`、`kind`、`shape`、`dtype`、`layout`、`edge`、`producer`、
`consumer`、`t_shape`、`t_dtype`、`out_shape`、`out_dtype`、`logical_save_bytes` 及其
写/读拆分、`seeds`、`tolerance`、`fp64_reference`、`expected_zero_output`、`planned_status`。

两个试点：`P1-n65536-f32-contig-t`、`R1-m256-k512-f32-contig-t`。
