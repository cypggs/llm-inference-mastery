# LLM 推理服务学习总计划

> 基于 [patchy631/time-to-first-token](repo/README.md) 10 周路线图，结合本机条件（compshare 已有一台停止状态的 RTX 5090 32GB 实例 `cpo...`，可随时 `compshare instance start` 启用）定制的可执行计划。

## 核心原则

1. **一个服务贯穿始终**：所有实验都作用在同一个 OpenAI 兼容推理服务上，不做 17 个互不相关的实验。
2. **先测量后优化**：Week 3 的监控 + Week 5 的压测是所有后续优化的前提。
3. **GPU 按需租用**：只在 build 日开机，用完 `compshare instance stop`。每周开始前批量规划租机。
4. **每周产出笔记**：`notes/week-XX-*.md`，可查阅可复习。

## 目标模型与硬件

| 项 | 选择 | 理由 |
|---|---|---|
| 模型 | **Qwen2.5-7B-Instruct**（FP16 约 15GB） | 中文友好，单卡 32GB 充裕，社区量化变体齐全 |
| GPU | compshare RTX 5090 32GB（已有实例） | 覆盖 Week 2/3/4/6/7 全部 build |
| 高配 | Week 5 千并发压测需要 H100，届时在 compshare 单独租 1 台，用完即删 |
| Week 8 | 优先 `kind` 本地集群 + 单机 Docker 兜底，避免集群管道工吃掉学习时间 |

## 10 周节奏总览

| 周 | 主题 | 需要 GPU? | 笔记 |
|---|---|---|---|
| 1 | 心智模型：Roofline、算术强度、prefill vs decode | ❌ 纯阅读+手算 | [week-01](notes/week-01-mental-model.md) |
| 2 | vLLM：部署 + 读 PagedAttention/V1 调度器源码 | ✅ Tue/Fri 各 ~1h | week-02 |
| 3 | 测量：Prometheus + Grafana 监控栈 | ✅ Tue/Wed | week-03 |
| 4 | SGLang：RadixAttention、连续批处理、chunked prefill | ✅ Tue + Buffer | week-04 |
| 5 | 压测到 1000 并发：GuideLLM / vllm bench / genai-perf | ✅ Tue–Fri，**Fri 需 H100** | week-05 |
| 6 | 量化：FP8 / INT4(AWQ/GPTQ) + 质量对照 | ✅ Thu/Fri | week-06 |
| 7 | 投机解码 + KV 驱逐 | ✅ Wed/Fri | week-07 |
| 8 | 分离式服务（PD disaggregation）+ K8s 自动扩缩 | ✅ Thu/Fri（可单机模拟） | week-08 |
| 9 | 单位经济学 + 成本路由 | ✅ Tue（复用 W5 数据可不开机） | week-09 |
| 10 | 发布可复现 benchmark + 建立阅读习惯 | ❌ 写作 | week-10 |

## compshare GPU 操作速查

```bash
# 列出实例（已有 5090 实例）
compshare instance list

# 开机（把 <id> 换成实际实例 ID）
compshare instance start <id>

# 等实例就绪
compshare instance wait <id> --state Running

# 获取 SSH 命令
compshare instance ssh <id> --show-sensitive

# 用完立即关机（停止计费）
compshare instance stop <id>

# Week 5 临时租 H100 前先询价
compshare instance price --gpu H100 --count 1 --cpu 16 --memory 128GiB --region <r> --zone <z>
```

> ⚠️ 纪律：每次 GPU session 结束 = 保存数据/笔记 → `instance stop` → 确认状态为 Stopped。

## 发布前 Checklist（Week 5 读，Week 10 用）

见 [repo/progress.md](repo/progress.md) 末尾 8 条。关键三条：
- 扫请求速率而不是测单一并发点；报告拐点不报告崩溃点
- TTFT 和 ITL 分开报告；p50/p95/p99 都要
- 钉死引擎版本 + 公布完整命令

## 进度追踪

本目录下的 [progress.md](progress.md) 是本地追踪副本，每完成一节课打勾并在笔记里写产出。
