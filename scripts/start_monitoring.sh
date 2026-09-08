#!/bin/bash
# ============================================
# Week 3 监控栈一键启动脚本
# 启动/检查：vLLM (:8000) + Prometheus (:9090) + Grafana (:3000)
# 用法: bash /workspace/start_monitoring.sh
# ============================================
set -u

VLLM_BIN="/usr/local/miniconda3/envs/py312/bin/vllm"
MODEL_PATH="/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
MON_DIR="/opt/monitoring"

echo "======================================"
echo " 推理服务监控栈 一键启动"
echo "======================================"

# ---------- 1. vLLM ----------
if curl -s -m 3 http://localhost:8000/v1/models | grep -q Qwen; then
    echo "✅ vLLM 已在运行 (:8000)"
else
    echo "🚀 启动 vLLM ..."
    cd /workspace
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    nohup $VLLM_BIN serve "$MODEL_PATH" \
        --served-model-name Qwen2.5-7B-Instruct \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --port 8000 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        > /workspace/vllm.log 2>&1 &
    disown
    echo "   启动中（约 2-3 分钟），日志: /workspace/vllm.log"
fi

# ---------- 2. Prometheus ----------
if curl -s -m 3 http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus 已在运行 (:9090)"
else
    echo "🚀 启动 Prometheus ..."
    cd $MON_DIR/prometheus
    nohup ./prometheus \
        --config.file=prometheus.yml \
        --storage.tsdb.path=$MON_DIR/prom-data \
        --web.listen-address=0.0.0.0:9090 \
        > $MON_DIR/prometheus.log 2>&1 &
    disown
    sleep 2
    echo "   已启动"
fi

# ---------- 3. Grafana ----------
if curl -s -m 3 http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana 已在运行 (:3000)"
else
    echo "🚀 启动 Grafana ..."
    cd $MON_DIR/grafana
    nohup ./bin/grafana server \
        --homepath $MON_DIR/grafana \
        cfg:server.http_addr=0.0.0.0 cfg:server.http_port=3000 \
        > $MON_DIR/grafana.log 2>&1 &
    disown
    sleep 2
    echo "   已启动"
fi

echo "======================================"
echo " 访问入口"
echo "======================================"
HOST_PATTERN="<实例ID>.pod.compshare.cn"
echo "  📓 JupyterLab  :8888  (https://8888-${HOST_PATTERN})"
echo "  📊 Grafana     :3000  admin/admin"
echo "  🔍 Prometheus  :9090"
echo "  🤖 vLLM API    :8000"
echo ""
echo "  Dashboard: Grafana → Dashboards → vLLM"
echo "======================================"
