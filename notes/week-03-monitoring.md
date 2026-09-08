# Week 3 · 监控篇：给推理服务装上仪表盘

> 🎯 核心思想：**先测量，后优化**。Week 6/7 的所有"性能提升"都只能靠本周的监控来证明。
> ✅ 2026-09-07 已在实例 `cpod-1uzuvryhbo2m` 完成部署（二进制方式，无 Docker）。

## 0. 监控体系架构

```
vLLM 服务 (:8000)
   └─ /metrics 端点（Prometheus 格式文本，98 个 vllm:* 指标）
        ↑ 每 5 秒抓取
Prometheus (:9090)          ← 时序数据库，存所有指标的历史曲线
   ↑ 查询（PromQL）
Grafana (:3000)             ← 可视化，12 面板的官方 vLLM Dashboard
   ↑ 浏览器访问
你
```

## 1. 四个必懂指标（Week 3 的灵魂）

| 指标 | 含义 | 对应 Week 1 概念 |
|---|---|---|
| `vllm:time_to_first_token_seconds` (histogram) | TTFT 首字延迟 | prefill 速度（算力瓶颈） |
| `vllm:inter_token_latency_seconds` (histogram) | ITL 字间延迟 | decode 速度（带宽瓶颈） |
| `vllm:num_requests_running` / `waiting` | 运行中/排队中的请求数 | 连续批处理的实时状态 |
| `vllm:kv_cache_usage_perc` | KV Cache 池使用率 | 显存利用率，接近 100% → 抢占 |

> 🔑 判读口诀：**waiting 涨 + TTFT 涨 = 排队了；running 不动 + kv_cache_usage 满 = 要抢占了**（Week 5 压测时回来验证）。

## 2. 本次部署实录（二进制方式，容器环境无 Docker）

```bash
# Prometheus v3.14.0
/opt/monitoring/prometheus/prometheus \
  --config.file=/opt/monitoring/prometheus/prometheus.yml \
  --storage.tsdb.path=/opt/monitoring/prom-data \
  --web.listen-address=0.0.0.0:9090

# prometheus.yml 核心配置
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["localhost:8000"]     # 每 5s 抓 vLLM 的 /metrics

# Grafana 13.2.1（注意：v13 二进制改名为 grafana server，不再是 grafana-server）
/opt/monitoring/grafana/bin/grafana server \
  --homepath /opt/monitoring/grafana \
  cfg:server.http_addr=0.0.0.0 cfg:server.http_port=3000
```

**踩坑记录**：
1. 455MB 的 Grafana 包国内下载慢（~10 分钟），且 `curl -C -` 断点续传 + 重定向会损坏文件 → 必须完整重下
2. Grafana 13 启动命令从 `grafana-server` 变为 `grafana server`
3. Dashboard 导入：去掉 JSON 里硬编码的 `id` 字段，POST 到 `/api/dashboards/db`

## 3. 访问入口

| 服务 | 端口 | 公网 URL（端口映射规则） |
|---|---|---|
| **Grafana** | 3000 | `https://3000-cpod-1uzuvryhbo2m-s1.pod.compshare.cn` （admin/admin）|
| Prometheus | 9090 | `https://9090-cpod-1uzuvryhbo2m-s1.pod.compshare.cn` |
| vLLM API | 8000 | （建议保持内网，压测用） |
| JupyterLab | 8888 | 已在用 |

Grafana 里打开 **Dashboards → vLLM**：TTFT、ITL、Token 吞吐、Scheduler 状态、KV Cache 利用率、排队时间等 12 个面板。

## 4. 课后实验（在 JupyterLab 里做）

1. 跑 `bench.py` 的 64 并发，盯着 **Scheduler State** 面板看 running 请求数爬上 64
2. 观察 **Cache Utilization**：64 并发 × 256 token 占 KV 池的比例（对照 Week 1 手算）
3. 改 `--max-num-seqs 8` 重启 vLLM 再压测，看 **waiting 队列**第一次暴涨——这就是 Week 5 要找的"拐点"

## 资源
- [vLLM metrics 设计文档](https://docs.vllm.ai/en/latest/design/v1/metrics/)
- [官方 Prometheus+Grafana 示例](https://github.com/vllm-project/vllm/tree/main/examples/observability/prometheus_grafana)
- [Modal: How to Benchmark LLM Engines](https://modal.com/llm-almanac/how-to-benchmark)（周五阅读）
