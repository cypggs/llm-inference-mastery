# 本地进度追踪

开始日期：2026-09-07

## Week 1: The mental model（无需 GPU）
- [x] Mon. Making Deep Learning Go Brrrr 前半（compute/memory/overhead 三 regime）
- [x] Tue. Making Deep Learning Go Brrrr 后半（算子融合、overhead、regime 判断法）
- [x] Wed. CS336 Lecture 5 GPUs 前 30min（执行模型、显存层级）— 文字版 kipply 前半
- [x] Thu. CS336 Lecture 5 后 30min（算术强度、roofline）— 文字版 kipply 后半
- [x] Fri. 手推 Qwen2.5-7B 在 RTX 5090 上的算术强度与拐点 batch size
- 笔记：[notes/week-01-mental-model.md](notes/week-01-mental-model.md)

## Week 2: vLLM
- [ ] Mon. PagedAttention 公告 + SOSP 论文 block-table 节
- [x] Tue. 【GPU】vLLM OpenAI 兼容服务部署 Qwen2.5-7B ✅ 2026-09-07 实机完成：5090，vLLM 0.25.1，KV cache 253k token，单流 104 tok/s，64并发总吞吐 5317 tok/s（详见笔记 §4.5）
- [ ] Wed. vLLM V1 架构（Office Hours 22 + blog）
- [ ] Thu. Inside vLLM 解剖：scheduler + block manager
- [ ] Fri. 【GPU 可选】跟读 block table 代码路径，写 3 句话解释块查找

## Week 3: Measurement
- [x] Mon. vLLM metrics 设计文档（笔记已覆盖四大核心指标）
- [x] Tue. 【GPU】Prometheus + Grafana 栈 ✅ 二进制部署完成（无 Docker）
- [x] Wed. 【GPU】导入官方 dashboard（12 面板），压测数据实时可见
- [x] Thu. Databricks + NVIDIA 指标映射（笔记 §5 面板↔指标对照表）
- [x] Fri. Modal 基准方法论（Notebook §3 P95/P99 思想已覆盖）
- 笔记：[notes/week-03-monitoring.md](notes/week-03-monitoring.md) + 实例上交互 Notebook

## Week 4: SGLang
- [x] Mon. RadixAttention blog（笔记 §1）
- [x] Tue. 【GPU】同模型同硬件部署 SGLang ✅ 0.5.19 与 vLLM 0.28 共存
- [x] Wed. SGLang talk + NeurIPS 论文（笔记 §1.3 对比表）
- [x] Thu. 连续批处理 + Orca（笔记 §2 甘特图）
- [x] Fri. Sarathi-Serve chunked prefill（笔记 §3）
- [x] Buffer. 【GPU】vLLM vs SGLang 前缀对决 ✅ 3360token 共享前缀，TTFT 冷/热数据已记录
- 笔记：[notes/week-04-sglang-batching.md](notes/week-04-sglang-batching.md)

## Week 5: Load testing
- [x] Mon. GuideLLM 介绍 + 安装
- [x] Tue. 【GPU】GuideLLM sweep 扫描 ✅ 找到开环拐点 ~20 RPS
- [x] Wed. 【GPU】vllm bench serve 交叉验证 ✅ 两工具差异 3-13% 全部可解释
- [x] Thu. 【GPU】并发扫描 1→128 ✅ 闭环拐点 ~128 并发（genai-perf 需容器环境，用等效自研脚本）
- [ ] Fri. 【GPU·H100】冲 1000+ 并发，观察 KV 利用率与抢占（待租机）
- 笔记：[notes/week-05-load-testing.md](notes/week-05-load-testing.md)

## Week 6: Quantization
- [x] Mon. MIT 6.5940 Lecture 5 前半（笔记 §1 覆盖）
- [x] Tue. Lecture 5 后半 + Lecture 6（笔记 §2 方法地图覆盖 AWQ/GPTQ/SmoothQuant）
- [x] Wed. Lilian Weng 综述量化节 + 引擎方法映射（笔记 §2，vLLM 支持 fp8/awq/gptq + fp8 KV）
- [x] Thu. 【GPU】FP8 在线量化部署 ✅ 单流 ×1.58，KV 池 +30%（AWQ 跳过，避免下载耗时）
- [x] Fri. 【GPU】质量对照 ✅ 6/6 正确，与 FP16 语义等价
- 笔记：[notes/week-06-quantization.md](notes/week-06-quantization.md)

## Week 7: Speculative decoding & KV eviction
- [x] Mon. 投机解码文档 + benchmark 文章（笔记 §1，交叉点）
- [x] Tue. GPU MODE Lecture 22 前半（笔记 §1.2 三步循环覆盖）
- [x] Wed. 【GPU】N-gram 投机解码实验 ✅ **负结果**：接受率 98.6% 仍全线减速（CPU overhead 主导），交叉点 < QPS 1
- [x] Thu. StreamingLLM（笔记 §2 注意力汇聚）
- [ ] Fri. 【GPU】长上下文驱逐策略，测内存与 TTFT（可选，未做）
- 笔记：[notes/week-07-speculative-kv.md](notes/week-07-speculative-kv.md)

## Week 8: Disaggregation & K8s
- [ ] Mon. DistServe talk + 论文
- [ ] Tue. Splitwise + vLLM disagg prefill 文档
- [ ] Wed. Mooncake + SGLang PD 分离
- [ ] Thu. 【GPU/kind】production-stack Helm 部署
- [ ] Fri. 【GPU/kind】按队列深度（非 CPU）自动扩缩

## Week 9: Economics & Router
- [x] Mon. 推理经济学第一性原理（笔记 §0-1，利用率 > 单价）
- [x] Tue. 用自己的 W5 吞吐数据推 $/1M tokens ✅ 拐点 2500 tok/s → 满产 ¥0.36/1M，30% 利用率 → ¥1.19/1M
- [x] Wed. Character.AI 33x 降本（笔记 §2 杠杆对照表）
- [x] Thu. 【写码】成本/延迟/质量路由器 ✅ [router/router.py](router/router.py)：分级+预算闸门+成本账本
- [x] Fri. 【写码】token 预算中间件（已并入路由器，待开机联调上 dashboard）
- 笔记：[notes/week-09-economics-router.md](notes/week-09-economics-router.md)

## Week 10: Publish
- [ ] Mon. 起草 writeup，过 checklist
- [ ] Tue. 补变体、钉版本、发布
- [ ] Wed. 建立每周阅读习惯
- [ ] Thu. 第一次习惯 session
- [ ] Fri. （可选）边缘端 sampler
