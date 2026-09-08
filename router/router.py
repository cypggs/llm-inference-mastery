#!/usr/bin/env python3
"""Week 9 交付物：成本/延迟/质量路由器
把 OpenAI 格式的请求按策略路由到不同后端，并记录每次决策的成本。
运行：pip install fastapi uvicorn httpx && python3 router.py
"""
import time, json, asyncio
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# ---------- 后端注册表（成本数据来自 Week 5/6 实测）----------
BACKENDS = {
    "local-7b": {
        "url": "http://localhost:8000/v1",          # vLLM 本地 7B
        "cost_in": 0.0002,   # 元/1K输入（自算：3.21元/h ÷ 实测吞吐）
        "cost_out": 0.0004,  # 元/1K输出
        "quality": 1,        # 质量等级（1=基础）
    },
    "local-7b-fp8": {
        "url": "http://localhost:8001/v1",          # FP8 变体（更快更便宜）
        "cost_in": 0.00013,
        "cost_out": 0.00027,
        "quality": 1,
    },
    # "api-large": {
    #     "url": "https://api.example.com/v1",      # 贵重大模型
    #     "cost_in": 0.002, "cost_out": 0.02, "quality": 3,
    # },
}

# ---------- Token 预算（成本守门员）----------
BUDGET = {"max_input_tokens": 4096, "max_output_tokens": 2048,
          "max_cost_per_request": 0.05}  # 元

# ---------- 成本账本（生产应写 Prometheus，这里演示用内存）----------
ledger = {"requests": [], "total_cost": 0.0}

app = FastAPI(title="Week9 Cost Router")

def classify_complexity(messages: list) -> str:
    """启发式分级：判断请求该去便宜端还是贵端。
    生产可换成：用小模型做难度分类（成本可忽略）。"""
    text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    hard_signals = ["代码", "code", "证明", "推导", "分析", "debug", "架构", "数学", "论文"]
    if len(text) > 1500 or any(k in text for k in hard_signals):
        return "hard"
    return "easy"

def pick_backend(complexity: str) -> str:
    """路由策略：简单请求走最便宜的，复杂请求走质量最好的。"""
    if complexity == "easy":
        return min(BACKENDS, key=lambda b: BACKENDS[b]["cost_out"])   # local-7b-fp8
    return max(BACKENDS, key=lambda b: BACKENDS[b]["quality"])        # 质量最高

def estimate_tokens(messages: list) -> int:
    return sum(len(str(m.get("content", ""))) // 2 for m in messages)  # 粗估

@app.post("/v1/chat/completions")
async def route(req: Request):
    body = await req.json()
    messages = body.get("messages", [])

    # 1. Token 预算检查
    est_in = estimate_tokens(messages)
    max_out = min(body.get("max_tokens", 512), BUDGET["max_output_tokens"])
    if est_in > BUDGET["max_input_tokens"]:
        return JSONResponse({"error": f"输入超预算（{est_in}>{BUDGET['max_input_tokens']} tokens）"},
                            status_code=413)

    # 2. 路由决策
    complexity = classify_complexity(messages)
    backend_name = pick_backend(complexity)
    backend = BACKENDS[backend_name]

    # 3. 成本预估与预算闸门
    est_cost = (est_in * backend["cost_in"] + max_out * backend["cost_out"]) / 1000
    if est_cost > BUDGET["max_cost_per_request"]:
        return JSONResponse({"error": f"预估成本 ¥{est_cost:.4f} 超单请求预算"}, status_code=429)

    # 4. 转发到选中后端
    body["max_tokens"] = max_out
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{backend['url']}/chat/completions", json=body)
    resp = r.json()
    latency = time.perf_counter() - t0

    # 5. 记账（真实 tokens 来自后端 usage）
    usage = resp.get("usage", {})
    real_cost = (usage.get("prompt_tokens", 0) * backend["cost_in"]
                 + usage.get("completion_tokens", 0) * backend["cost_out"]) / 1000
    ledger["total_cost"] += real_cost
    ledger["requests"].append({
        "complexity": complexity, "backend": backend_name,
        "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
        "cost": round(real_cost, 6), "latency_s": round(latency, 2),
    })
    print(f"[路由] {complexity:4} → {backend_name:12} | "
          f"{usage.get('prompt_tokens',0)}+{usage.get('completion_tokens',0)} tok | "
          f"¥{real_cost:.5f} | {latency:.1f}s")
    return resp

@app.get("/stats")
async def stats():
    """成本看板：累计花费、按后端分布、平均成本。"""
    reqs = ledger["requests"]
    by_backend = {}
    for r in reqs:
        b = by_backend.setdefault(r["backend"], {"n": 0, "cost": 0.0})
        b["n"] += 1; b["cost"] += r["cost"]
    return {"total_requests": len(reqs),
            "total_cost_yuan": round(ledger["total_cost"], 4),
            "avg_cost_yuan": round(ledger["total_cost"] / max(len(reqs), 1), 6),
            "by_backend": by_backend,
            "recent": reqs[-10:]}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Week9 成本路由器启动于 :9000")
    print("   转发 OpenAI 格式请求到本地后端，按复杂度选路，记录每分钱")
    uvicorn.run(app, host="0.0.0.0", port=9000)
