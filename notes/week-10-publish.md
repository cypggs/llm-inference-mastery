# Week 10 · 发布篇：把十周变成一份"可辩护"的作品

> 🎯 **本周目标**：① 把九周成果整合成可复现的 benchmark 报告并通过发布 checklist；② 建立让知识不过时的每周阅读习惯。
> 十周的终点不是"学完了"，而是"有证据"。

---

## 0. 我们的发布资产盘点（真实数据，全部可复现）

| 资产 | 内容 | 出处 |
|---|---|---|
| 硬件指纹 | RTX 5090 32GB · vLLM 0.28 · torch 2.13 · Qwen2.5-7B | 钉死版本 ✅ |
| 理论验证 | 手算 decode 上限 118 tok/s vs 实测 104（88%）；KV 池手算 vs 实测误差 0.7% | W1/W2 |
| 压测曲线 | 开环拐点 ~20 RPS（TTFT 50ms→92ms 起跳）；饱和区 27 RPS TTFT 8.2s 标记"不可发布" | W5 |
| 交叉验证 | GuideLLM vs vllm bench：TTFT 差 3% | W5 |
| 双引擎对决 | 3360 token 共享前缀：vLLM 冷 747ms vs SGLang 冷 2136ms；热轮打平 | W4 |
| 量化对照 | FP8：单流 ×1.58、KV 池 +30%、6/6 质量正确 | W6 |
| 负结果 | N-gram 投机解码接受率 98.6% 仍全线减速（CPU overhead 主导） | W7 |
| 经济学 | 满产 ¥0.36/1M tokens，30% 利用率 ¥1.19/1M | W9 |
| 成本路由器 | router.py：分级 + 预算闸门 + 成本账本 | W9 |

**这套数据的独特价值**：不是抄来的数字，是同一个硬件上理论→预测→实测→修正的完整闭环。

## 1. 发布 Checklist 逐条过堂（路线图原题）

- [x] **扫了请求速率**：GuideLLM 7 档 + 并发 5 档，无单点轶事
- [x] **报告了精确 token 长度**：256 输入 / 128 输出，每次实验标注
- [x] **报了 P50/P95/P99**：TTFT 报 P99，ITL 报 P50
- [x] **TTFT 和 ITL 分开**：所有表格两列独立
- [x] **未在抢占区采数**：27 RPS 饱和档明确标注"不可发布"
- [x] **长度分布说明**：固定长度是刻意的（第一周压测），已在文中声明
- [x] **双工具交叉验证**：3% 差异已解释
- [x] **钉死版本+命令**：环境指纹 + 全部复现脚本（scripts/）

## 2. 每周阅读习惯（让知识不过时的系统）

十周学到的具体数字会过时（就像 SGLang 的 5× 已被 vLLM V1 追平），**保持更新的能力比存量知识值钱**：

```mermaid
flowchart LR
    A["每周 30 分钟"] --> B["扫标题 10min<br/>arXiv cs.DC + MLSys/OSDI"]
    B --> C["深读一篇 15min<br/>只看：摘要+方法图+评测"]
    C --> D["记三行 5min<br/>问题→方法→对你的意义"]
```

**信息源分层**：
- 必读：[vLLM blog](https://blog.vllm.ai) · [LMSYS blog](https://lmsys.org/blog/) · [Modal Almanac](https://modal.com/llm-almanac/summary)
- 会议：MLSys / OSDI / NSDI / ATC（系统论文，作者亲自讲）
- 跳过：模型发布的公关稿（读系统论文，不读新闻稿）

## 3. 十周全景回顾（一张图带走整个课程）

| 周 | 你获得的超能力 | 关键数字 |
|---|---|---|
| 1 | 用 roofline 预判任何优化是否有效 | 拐点 234 FLOP/B |
| 2 | 部署并读懂 vLLM | KV 池 253,056 token |
| 3 | 给服务装仪表盘 | 98 个 vllm 指标 |
| 4 | 拆解任何引擎的四问框架 | vLLM vs SGLang 747ms vs 2136ms |
| 5 | 做出能辩护的压测 | 拐点 20 RPS，饱和 TTFT 8.2s |
| 6 | 量化三笔账 | FP8 ×1.58 无损 |
| 7 | 识破伪优化（负结果） | 98.6% 接受率仍减速 |
| 8 | 设计会呼吸的架构 | 扩缩信号 = 队列深度 |
| 9 | 把技术翻译成钱 | ¥0.36→3.57/1M 全看利用率 |
| 10 | **发布与持续进化** | 本仓库 |

## 4. 原始学习资料

- 📄 [arXiv cs.DC](https://arxiv.org/list/cs.DC/recent) · [MLSys](https://mlsys.org) · [USENIX](https://www.usenix.org/conferences)
- 📄 [vLLM blog](https://blog.vllm.ai) · [LMSYS blog](https://lmsys.org/blog/) · [Modal Almanac](https://modal.com/llm-almanac/summary)
- 🎬 [vLLM Office Hours](https://www.youtube.com/@vllm-project) · [GPU MODE](https://www.youtube.com/@GPUMODE)
