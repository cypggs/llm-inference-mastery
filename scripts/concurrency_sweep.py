#!/usr/bin/env python3
"""Week 5 实验三：闭环并发扫描找拐点（替代 genai-perf 的等效实验）
并发梯度 1/8/32/64/128，固定 256 输入 + 128 输出 token
"""
import json, time, urllib.request, threading, statistics, csv

URL = "http://localhost:8000/v1/completions"
PROMPT = "请详细介绍一下中国历史。" * 32  # ~256 token

def one(result, idx):
    payload = {"model": "Qwen2.5-7B-Instruct", "prompt": PROMPT,
               "max_tokens": 128, "temperature": 0, "stream": True}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.startswith(b"data: ") or b"[DONE]" in line: continue
                d = json.loads(line[6:])["choices"][0].get("text")
                if d:
                    if ttft is None: ttft = time.perf_counter() - t0
                    tokens += 1
        total = time.perf_counter() - t0
        result[idx] = {"ttft_ms": ttft * 1000, "tokens": tokens,
                       "itl_ms": (total - ttft) / max(tokens - 1, 1) * 1000}
    except Exception as e:
        result[idx] = None

def run(conc):
    results = [None] * conc
    ts = [threading.Thread(target=one, args=(results, i)) for i in range(conc)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    ok = [r for r in results if r]
    out_tps = sum(r["tokens"] for r in ok) / wall
    print(f"并发 {conc:>4}: 输出 {out_tps:7.0f} tok/s | "
          f"TTFT p50={statistics.median(r['ttft_ms'] for r in ok):6.0f}ms "
          f"p99={sorted(r['ttft_ms'] for r in ok)[int(len(ok)*0.99)-1]:6.0f}ms | "
          f"ITL p50={statistics.median(r['itl_ms'] for r in ok):5.1f}ms | "
          f"成功率 {len(ok)}/{conc}")
    return {"conc": conc, "out_tps": out_tps,
            "ttft_p50": statistics.median(r['ttft_ms'] for r in ok),
            "itl_p50": statistics.median(r['itl_ms'] for r in ok),
            "success": f"{len(ok)}/{conc}"}

rows = [run(c) for c in [1, 8, 32, 64, 128]]
with open("/workspace/concurrency_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("\n已保存 /workspace/concurrency_sweep.csv")
