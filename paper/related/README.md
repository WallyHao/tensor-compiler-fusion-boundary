# 相关工作语料

2026-11-28 下载到本目录的 PDF，用于完善 `paper/report.typ` 的相关工作、预测节和讨论。
PDF 本身不入库（见 `.gitignore`），本表是索引；按 DOI/arXiv 链接可重新获取。
每篇均核对了首页标题与文件签名（`%PDF`），未使用关键词猜名。

## 研究论文

| 文件 | 论文 | 发表 | 与本文的关系 |
|---|---|---|---|
| `pytorch2-asplos24.pdf` | PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation | ASPLOS 2024 (ACM), pp. 929–947, DOI 10.1145/3620665.3640366 | 实验对象 TorchInductor 的设计与调度器上下文；正文已引 |
| `triton-mapl19.pdf` | Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | MAPL 2019 (ACM), pp. 10–19, DOI 10.1145/3315508.3329973 | 生成 kernel 的 tile IR；`XBLOCK`/`RBLOCK`/warps 语义的来源 |
| `dnnfusion-pldi21.pdf` | DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion | PLDI 2021 (ACM), pp. 883–898, DOI 10.1145/3453483.3454083 | 融合方案枚举与代价模型；把融合选择当搜索问题 |
| `astitch-asplos22.pdf` | AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures | ASPLOS 2022 (ACM), pp. 359–373, DOI 10.1145/3503222.3507723 | SIMT 上内存密集算子的垂直/水平融合与 stitching；与本文压力家族最接近的融合空间工作 |
| `hfuse-cgo22.pdf` | Automatic Horizontal Fusion for GPU Kernels | CGO 2022 (IEEE/ACM), pp. 14–27, DOI 10.1109/CGO53902.2022.9741270 | 水平融合；补足本文只测垂直单消费者边 |
| `hidet-asplos23.pdf` | Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs | ASPLOS 2023 (ACM), pp. 370–384, DOI 10.1145/3575693.3575702 | 调度后自动融合（post-scheduling fusion）与任务映射 |
| `korch-asplos24.pdf` | Optimal Kernel Orchestration for Tensor Programs with Korch | ASPLOS 2024 (ACM), pp. 755–769, DOI 10.1145/3620666.3651383 | 内核编排的最优性；把融合/切分形式化为优化问题 |
| `neptune-pldi26.pdf` | Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs | Proc. ACM Program. Lang. (PLDI) 2026, pp. 1278–1302, DOI 10.1145/3808298 | GPU 算子融合中 locality 与 parallelism 的权衡 |
| `redfuser-arxiv26.pdf` | RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators | arXiv:2603.10026 (2026) | 级联归约的自动融合；与“epilogue 继承归约 tile”直接对应 |
| `megakernel-arxiv26.pdf` | One Megakernel is Not Enough: Predicting Where to Cut for Streaming Inference | arXiv:2610.35336 (2026) | 流式推理的切分预测；正文已引 |

## 对照分析

- `style-structure-comparison.md`：手稿与 DNNFusion (PLDI'21)、AStitch (ASPLOS'22)、Korch (ASPLOS'24) 的章节结构、篇幅分配、行文风格与表格使用对照；含 P0–P2 修改清单。
- `figure-comparison.md`：10 张图/2 张表与三篇论文的图法对照；记录两处硬缺陷（device 对角线 71/72、inconclusive 术语）及机制图、决策图、绝对时延图的实现方案。

## 版式参照

| 文件 | 内容 | 说明 |
|---|---|---|
| `acmsmall-sample.pdf` | ACM `acmart` 官方 `acmsmall` 样例（CTAN, 2026-11-25） | 与本文 `paper/template/approximate-acmsmall.typ` 同为 ACM 单栏样式，用于对照排版；不是研究论文 |

## 未纳入

- **Automated Kernel Fusion for GPU Based on Code Motion**（LCTES 2022, DOI 10.1145/3519941.3535078）：CC-BY 开放获取，但 ACM DL 的 PDF 端面对脚本请求返回 403，需浏览器手动下载。
- **User-driven Online Kernel Fusion for SYCL**（TACO 2023, DOI 10.1145/3571284）：acmsmall 期刊论文样例；同样受 ACM DL 脚本访问限制。

## 重新获取

```bash
# arXiv 版本
curl -L https://arxiv.org/pdf/2108.13342 -o dnnfusion-pldi21.pdf
# ACM 版本按 DOI 在浏览器打开后下载
```

许可：arXiv 论文遵循 arXiv 许可，ACM 论文按 ACM 条款；此处仅为本地阅读副本。
