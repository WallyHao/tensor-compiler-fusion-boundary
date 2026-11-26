# 02 · 采集与复跑

## 环境与顺序

1. 在 RTX 3090 主机核对 GPU 名称、UUID、驱动、功耗和时钟策略以及持久盘。使用 `requirements.lock` 对应的 PyTorch 2.6.0+cu124、CUDA 12.4、Triton 3.2.0 和 Python 3.12，优先复用原 Python 3.12.15。驱动或 Python 补丁有变化时记录，并用同期对照检查差异。保存 commit、命令、锁文件哈希、`pip freeze` 及 GPU 快照。[环境操作手册](../../docs/3090-runbook.md)
2. 通过 [cases.csv](cases.csv) 生成固定清单，先验收三个 `pilot=yes` 配置的候选证据。试点每配置每模式取 5 个配对 block。三例两模式均有效时应收到 30 对。某组试点无法配对时，至多改用该组另一个已登记 ID 验证流程，保留首例失败与替代原因；组内仍无法配对时检查其余组，按实际覆盖交付。
3. 用独立 `run_id` 记录试点和正式数据。正式阶段重新编译或加载已核验的正式候选，测量自己的 L_direct、L_replay 与复制带宽。试点用于检查流程，正式分析使用新 run 的采集行。
4. 若评估原模型，用归档中的 RTX 3090 源模型文件和独立 split-only 采样生成冻结预测，记录模型、清单、校准、脚本和 split timing 的 SHA-256。预测覆盖包含新配置；保存 UTC 后才开始正式配对。试点结果只用于质量检查和确定采集能否运行。模型输入缺失则保留原因，配对继续。若 [冻结脚本](../../scripts/validate_transfer.py) 需要适配补采清单，沿用已验证的特征、公式及阈值。

## 正式采集与质量

使用现有 `python -m fb collect`，设备指定 `--expect-device "RTX 3090" --expect-uuid "$FB_EXPECT_UUID"`，候选目录由 `--provider` 指定，显式传 20 个 ID，使用 `--purpose full --calibrate --blocks 30 --modes direct,replay`。运行命令随实际实现交付。完整时每组分别有 240、480、240、240 个配对 block，总计 1,200 对、2,400 条合格 timing 行。排除或缺失以实际分母报告。

- 每 case/mode 固定调用批大小，30 个 block 中 SF/FS 各 15 个。使用 CUDA events 计时，协议参数与原 3090 run 一致。把 direct 作为主终点，replay 单独报告。保留所有 attempt，外部进程、节流变动、错误和缺半对按现有协议整对重试。编译、预热、捕获、校验及 profiler 分别记录，正式 timing 采用测量接口的批处理。
- 用实际 launch trace 核对采集执行的候选与编译清单对应。保存 `timings.jsonl`、`telemetry.jsonl`、`blocks/`、`state/`、`checks.jsonl`、变体证据及环境元数据。replay 无法捕获时标注具体错误，仍报告合格 direct。
- 计时全部完成后再诊断，先选最多 4 个配置。优先顺序为 direct 拆分占优、direct 默认错选、replay 拆分占优、replay 默认错选，全部没有时取各组有代表性的置信区间近零配置。组内同类按相对 regret 从大到小、同分按 case ID 排序。记录实际选择规则和缺失指标原因。能取到计数器时记录 kernel 数、寄存器、共享内存、spill、占用率、DRAM/L2 读写。
- 对最多 4 个上述配置用新 `run_id` 和事先记录的不同 seed 分别复跑 direct/replay，每模式 30 对。复跑使用同一设备、配对口径及候选证据，原 run 与复跑单独统计。收益符号不稳定时保留两份结果。若原固定配置不足以定位翻转区间，用现有相邻区间表达不确定性。

用 `fb verify` 查配对数、污染、单位、覆盖和恢复行为。采集补齐并完成诊断后用 `fb seal` 写 SHA-256 清单，在归档解包处复查。[配对采集手册](../../docs/measurement-runbook.md)
