# D · 数据契约与验收

目录以 runs/<run_id>/ 为根。建议结构化表用 JSONL，清单用 JSON，报告用 Markdown。所有路径相对 run 根目录。时间戳用 UTC ISO 8601，耗时用 µs（原始 CUDA event 另存 ms），容量用 bytes。数字保留采集精度，缺值用 null 并附 reason。run_id 区分正式、试点、复跑和 synthetic 数据。

| 文件 | 粒度与键（同一 run 内） | 必需内容 |
|---|---|---|
| run.json | 一次运行 | run_id、purpose、parent_run、协议/语料/代码哈希、命令、环境锁与 GPU 快照路径、起止时间、种子、负责人与状态 |
| cases.jsonl | case_id | template、shape、dtype、layout、edge、输入 seed 列表、t 的形状/类型、逻辑节省字节、计划状态和排除原因。建议 ID 如 R1-m256-k512-f32-contig-t |
| variants.jsonl | case_id, strategy | default/split/fused、编译配置、产物哈希与路径、边映射、是否物化、核验后的默认决策、compile_ms、kernel 列表及目标区域标记、编译期发射数/资源元数据、trace 路径与判定。失败也占一行 |
| checks.jsonl | case_id, strategy, input_id, mode | reference、容差、输出 shape/dtype、非有限数计数、最大误差、pass、reason |
| calibration.jsonl | mode, probe, size_or_k, block | buffer_bytes 或 k、repeats、elapsed_ms、顺序、派生单位、环境引用；另附校准系数和拟合脚本版本 |
| timings.jsonl | case_id, mode, block, attempt, strategy | iteration_order、n、elapsed_ms、us_per_call、时间窗、设备/stream、artifact_hash、quality、reason。mode 为 direct 或 replay，strategy 的配对值为 split/fused |
| telemetry.jsonl | timestamp, gpu_uuid | 温度、时钟、功率、节流状态、外部 GPU 活动；注明采样来源与不可读字段 |
| profiles.jsonl | case_id, strategy, kernel_index, metric, sample | metric 原名、来源、数值、单位、采集范围、trace/报告路径或缺测原因 |
| results.jsonl | case_id, mode | 候选/默认对应关系、有效对数、T_split/T_fused、B、相对收益、区间、标签、分组 fold、基线与模型预测、逐策略 regret、特征来源 |
| artifacts/ 与 freeze/ | 按 case/策略或文件名索引 | IR、代码、调度原因、trace、补丁、报告图表、分析入口、冻结参数及校验和 |

状态最少区分 planned、valid、unsupported、partition_mismatch、incorrect、compile_error、incomplete；将状态分配到相应 case/产物/模式，report 单列 direct 与 replay 的有效数。个别计数器缺失只影响对应诊断或特征。

**数据交接验算**

- 从登记清单左连接下游各表，核对 72 个唯一 case、各模板计划数 12/12/24/24、失败行及原因。统计各环节分母，报告因切边失败造成的样本选择偏差。
- 对每个有效 case/mode，筛出首个合格 attempt 的整对；检查 30 个不同 block、每对各有 split/fused 一行、n 相同且大于零、产物哈希与策略相符。由 elapsed_ms 重算 us_per_call；核对顺序平衡、时间戳及正确性检查。
- 验证主表之间的外键、证据文件存在且哈希一致。每个聚合数均能追溯到选用的原始键；独立 default、校准和 profiler 数据有各自标记，避免混入配对。
- 附一条新 run 复跑命令和一条只读原始文件生成表图的命令，写明锁文件、工作目录及参数。原始文件封存后派生结果写入独立目录；跨 run 合并始终加入 run_id。
