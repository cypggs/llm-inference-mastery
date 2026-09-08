#!/usr/bin/env python3
"""Week 4 实验：前缀复用对决 vLLM vs SGLang
用法: python3 prefix_faceoff.py --url http://localhost:8000 --name vLLM
"""
import argparse, json, time, urllib.request, threading, statistics

# 构造 ~2000+ token 的共享 system prompt（模拟客服/知识库场景）
BASE = """你是一家电商平台的智能客服助手。请严格遵守以下服务规范回答用户问题。

## 服务规范
一、退货政策：自签收之日起7天内可无理由退货，商品需保持完好，配件齐全。大家电退货需预约上门取件。
二、换货政策：15天内出现质量问题可免费换货，运费由平台承担。换货周期通常为3-5个工作日。
三、退款时效：审核通过后，余额支付实时到账，银行卡支付1-3个工作日，信用卡支付3-15个工作日。
四、发票规则：支持电子普通发票和增值税专用发票，下单时或签收后30天内可申请补开。
五、运费规则：普通商品满99元包邮，生鲜满59元包邮，偏远地区（新疆、西藏、内蒙古部分区域）需额外支付运费。
六、会员权益：PLUS会员享95折、每月运费券6张、专属客服通道、免费上门退换。
七、价保规则：自营商品30天内降价可申请价格保护，差价原路退回。
八、售后时效：工作日9:00-21:00在线，节假日9:00-18:00在线，紧急问题可拨打400热线。

## 回答风格要求
1. 先直接回答问题，再补充相关政策细节；2. 涉及金额、时效时给出具体数字；
3. 语气亲切专业，适当使用表情符号；4. 不确定的问题引导用户联系人工客服；
5. 每次回答结尾询问用户是否还有其他问题。

## 常见问题速查
- 修改地址：未发货可在订单页自行修改；已发货联系客服拦截。
- 优惠券：可在"我的-优惠券"查看，注意使用门槛和有效期。
- 账户安全：支持手机验证码、人脸识别双重验证。
- 物流查询：订单详情页实时更新，异常件48小时内处理。
- 商品真伪：自营商品均为品牌直供，支持专柜验货。
- 发票抬头：支持个人和企业抬头，企业需提供税号。
- 分期付款：支持花呗、白条、信用卡分期，最高24期免息。
- 赠品规则：赠品与主商品分开发货时，赠品物流单号会短信通知。
- 定制商品：不支持7天无理由退货，下单前请确认规格。
- 预售商品：定金支付后不支持退定金，尾款支付后可申请整单退款。
"""
SHARED_PREFIX = (BASE * 6)  # 约 2500-3000 token

QUESTIONS = [
    "我买的手机三天了还没发货怎么办？", "退货的运费谁承担？", "发票开错了能重开吗？",
    "PLUS会员怎么买最划算？", "新疆包邮吗？", "价保怎么申请？",
    "我的优惠券为什么不能叠加？", "定制了一个刻字水杯能退吗？",
    "信用卡退款多久到账？", "怎么查物流单号？",
] * 5  # 50 个请求

def chat(url, prompt, max_tokens=32):
    payload = {"model": "Qwen2.5-7B-Instruct",
               "messages": [{"role": "system", "content": SHARED_PREFIX},
                            {"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; tokens = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if d:
                if ttft is None: ttft = time.perf_counter() - t0
                tokens += 1
    return ttft, tokens

def run_round(url, concurrency, label):
    results = [None] * concurrency
    def w(i): results[i] = chat(url, QUESTIONS[i % len(QUESTIONS)])
    ts = [threading.Thread(target=w, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    ttfts = [r[0] * 1000 for r in results]
    toks = sum(r[1] for r in results)
    print(f"  [{label}] TTFT p50={statistics.median(ttfts):.0f}ms "
          f"max={max(ttfts):.0f}ms | 总吞吐={toks/wall:.0f} tok/s")
    return statistics.median(ttfts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--name", default="engine")
    args = ap.parse_args()

    # 先确认前缀的 token 数
    req = urllib.request.Request(f"{args.url}/v1/completions",
        data=json.dumps({"model": "Qwen2.5-7B-Instruct", "prompt": SHARED_PREFIX,
                         "max_tokens": 1}).encode(),
        headers={"Content-Type": "application/json"})
    usage = json.loads(urllib.request.urlopen(req).read())["usage"]
    print(f"\n===== {args.name} 对决测试 =====")
    print(f"共享前缀: {usage['prompt_tokens']} tokens\n")

    print("【冷缓存】第一轮 50 并发（引擎第一次见这个前缀）:")
    cold = run_round(args.url, 50, "冷")
    print("【热缓存】第二轮 50 并发（前缀应命中缓存）:")
    warm = run_round(args.url, 50, "热")
    print(f"\n>>> {args.name} 前缀复用收益: TTFT {cold:.0f}ms → {warm:.0f}ms "
          f"(提速 {cold/warm:.1f}x)\n")

if __name__ == "__main__":
    main()
