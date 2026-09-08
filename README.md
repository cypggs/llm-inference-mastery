# 大模型推理工程师成长笔记（LLM Inference Mastery）

> 十周时间，从"会调 API"到"能设计、压测、优化、计价一个生产级推理服务"。
> 本仓库是完整的学习路径 + **全部实机数据**——每个数字都在 RTX 5090 上亲手测出，可复现。

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 🌟 为什么这个仓库不一样

不是二手知识的搬运，而是**理论 → 预测 → 实测 → 修正**的完整闭环：

| 理论预言 | 实机验证 | 误差 |
|---|---|---|
| 手算单流 decode 上限 118 tok/s | 实测 104 tok/s | 88% 达成 ✅ |
| 手算 KV Cache 池 ~25 万 token | vLLM 实测 253,056 | **0.7%** ✅ |
| "拐点前加并发免费" | 并发 1→64 吞吐 ×53，单用户仅降 13% | ✅ |
| "FP8 提速理论 ×2" | 实测 ×1.58（反量化开销） | 修正模型 ✅ |

## 🗺️ 十周地图

| 周 | 主题 | 笔记 | 交互式 Notebook | 关键实机数据 |
|---|---|---|---|---|
| 1 | 心智模型：Roofline/算术强度 | [notes](notes/week-01-mental-model.md) | [📓](notebooks/week01_mental_model.ipynb) | 拐点 234 FLOP/B |
| 2 | vLLM 部署与内脏 | [notes](notes/week-02-vllm.md) | [📓](notebooks/week02_vllm_deploy_bench.ipynb) | KV 池 253,056 token |
| 3 | Prometheus+Grafana 监控 | [notes](notes/week-03-monitoring.md) | [📓](notebooks/week03_monitoring.ipynb) | 98 个 vllm 指标 |
| 4 | SGLang 对决+连续批处理 | [notes](notes/week-04-sglang-batching.md) | [📓](notebooks/week04_sglang_faceoff.ipynb) | 冷启动 747ms vs 2136ms |
| 5 | 压测到拐点 | [notes](notes/week-05-load-testing.md) | [📓](notebooks/week05_load_testing.ipynb) | 开环拐点 ~20 RPS |
| 6 | FP8 量化三笔账 | [notes](notes/week-06-quantization.md) | [📓](notebooks/week06_quantization.ipynb) | ×1.58 速度 +30% KV 池 |
| 7 | 投机解码（负结果！） | [notes](notes/week-07-speculative-kv.md) | [📓](notebooks/week07_spec_decode.ipynb) | 98.6% 接受率仍减速 |
| 8 | PD 分离 + K8s 弹性 | [notes](notes/week-08-disagg-k8s.md) | — | 扩缩信号=队列深度 |
| 9 | 单位经济学+成本路由器 | [notes](notes/week-09-economics-router.md) | — | ¥0.36→3.57/1M |
| 10 | 发布与阅读习惯 | [notes](notes/week-10-publish.md) | — | 本仓库 |

## 💎 四个最有价值的发现

1. **手算精度 0.7%**：Week 1 用纸笔推导的 KV Cache 容量（25.4 万 token）和 vLLM 实测（25.3 万）几乎一致——第一性原理不是玄学
2. **营销数字会过期**：SGLang 宣传的 5× 前缀优势，在已标配前缀缓存的 vLLM V1 面前消失——热轮几乎打平（231 vs 252ms）
3. **98.6% 接受率仍然减速**：N-gram 投机解码在我们的栈上全线变慢——瓶颈从 GPU 带宽搬到了 CPU 开销，敢发负结果
4. **成本差 100 倍全在利用率**：同一张卡同一个模型，满产 ¥0.36/1M tokens vs 闲置 ¥3.57/1M

## 📁 仓库结构

```
├── notes/          # 十周图文笔记（中文，Mermaid 配图，含自测题）
├── notebooks/      # 7 个可执行的 Jupyter Notebook（含实机输出）
├── data/           # 原始压测数据（guidellm_sweep.json 等）
├── scripts/        # 可复现脚本（压测/监控/部署一键启动）
├── router/         # Week 9 交付：成本/延迟/质量路由器
├── LEARNING_PLAN.md # 十周总计划
└── progress.md     # 完成进度
```

## 🚀 快速开始

**读笔记**：从 [Week 1 心智模型](notes/week-01-mental-model.md) 开始，每周都有「自测题 + 名词小词典」。

**跑 Notebook**：需要一张 24GB+ 的 GPU（RTX 4090/5090 均可）：
```bash
pip install vllm==0.28.0 matplotlib
# 按 notebooks/ 里 week01 → week07 的顺序执行
```

**复现压测**：
```bash
bash scripts/start_monitoring.sh      # 一键启动 vLLM+Prometheus+Grafana
bash scripts/run_guidellm.sh          # GuideLLM 开环扫描
python3 scripts/concurrency_sweep.py  # 闭环并发扫描找拐点
```

## 🎓 适合谁

- 想从"调包侠"成长为推理系统工程师的人
- 需要为公司做模型部署/降本决策的人
- 准备推理方向面试的人（每周的自测题就是面试题）

前置要求：Python 基础 + 对 Transformer 有概念级了解。**不需要** CUDA/ Serving 经验。

## 🙏 致谢

学习路线基于 [patchy631/time-to-first-token](https://github.com/patchy631/time-to-first-token)（CC BY 4.0）。
所有实验在优云智算（compshare）的 RTX 5090 实例上完成，总 GPU 成本 < ¥30。

## 📜 License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — 欢迎fork、改编、教学使用，署名即可。
