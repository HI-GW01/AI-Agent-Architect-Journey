# -*- coding: utf-8 -*-
"""
CoT 실험 — Direct(답만) vs Zero-shot CoT, 모델 4개 (크기 그라데이션), GSM8K 벤치마크

보고 싶은 것:
  1) 증류 모델(Flash-Lite)에서도 CoT 창발 효과가 보이나
  2) 최근 모델에도 CoT 방식이 여전히 효과적인가
  3) CoT 구현
  4) CoT에 따른 정답률 / 토큰 / 속도 차이

설계: 4모델 × 2조건(Direct / Step-by-step CoT) × N문제 (기본 50) = 400콜
  - few-shot 예시는 안 얹음(조건 폭발 방지). 전부 zero-shot.
  - 추론은 전 모델 OFF로 통일 → '프롬프트가 유도한 CoT'만 변수로 격리.
  - 데이터: GSM8K 원논문 벤치마크. 정답 검증되어 있어 채점 신뢰도 높음.

[실행]
1) pip install anthropic google-genai
2) 아래 두 API_KEY 채우기 (이전에 코드에 박혀 있던 키는 폐기하고 새로 발급할 것).
3) python CoT_test.py   (저장된 결과 재채점만: python rescore.py)
"""

import re
import csv
import json
import os
import random
import time
import urllib.request
from collections import defaultdict

# ============================================================
# 0. 설정
# ============================================================
ANTHROPIC_API_KEY = "Enter your key"
GEMINI_API_KEY    = "Enter your key"

# 모델 문자열은 placeholder. Anthropic console / Google AI Studio에서 정확한 문자열 확인 후 맞출 것.
MODELS = {
    # 라벨          (provider,  모델 문자열)
    "haiku":      ("claude",  "claude-haiku-4-5"),
    "sonnet":     ("claude",  "claude-sonnet-4-6"),
    "flash-lite": ("gemini",  "gemini-2.5-flash-lite"),
    "flash":      ("gemini",  "gemini-2.5-flash"),
}
# 출력 순서(작은→큰). 패밀리 안에서 규모 그라데이션이 눈에 보이게.
MODEL_ORDER = ["haiku", "flash-lite", "sonnet", "flash"]

# 단가 (100만 토큰당 USD) — 전부 placeholder. 각 provider 가격표에서 확인 후 수정.
# 라벨이 MODELS 키와 정확히 일치해야 함 (불일치 시 비용이 조용히 0으로 찍힘).
PRICE = {
    "haiku":      {"in": 1.00, "out": 5.00},
    "sonnet":     {"in": 3.00, "out": 15.00},
    "flash-lite": {"in": 0.10, "out": 0.40},
    "flash":      {"in": 0.30, "out": 2.50},
}

# ============================================================
# 1. GSM8K 로드 — 정답은 "####" 뒤 정수. 채점용이라 모델엔 안 줌.
# ============================================================
GSM8K_URL   = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
GSM8K_LOCAL = "gsm8k_test.jsonl"
N_PROBLEMS  = 50         # 30~50 사이에서 조절. 처음엔 30으로 파이프라인 확인 권장.


def load_gsm8k(n=N_PROBLEMS, seed=42):
    # 로컬 캐시 없으면 한 번 다운로드
    if not os.path.exists(GSM8K_LOCAL):
        print("Downloading GSM8K...")
        urllib.request.urlretrieve(GSM8K_URL, GSM8K_LOCAL)
    items = []
    with open(GSM8K_LOCAL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tail = d["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
            try:
                gold = int(tail)
            except ValueError:
                continue                      # 드문 비정수 답은 건너뜀
            items.append({"q": d["question"], "gold": gold})
    random.seed(seed)                          # 재현성: 매번 같은 표본
    sample = random.sample(items, min(n, len(items)))
    return [{"id": i + 1, "gold": s["gold"], "q": s["q"]} for i, s in enumerate(sample)]


PROBLEMS = load_gsm8k()

# ============================================================
# 2. 프롬프트 — 두 조건은 '지시문'만 다름 (출력 형식은 공통 → 변수 격리)
# ============================================================
PROMPT_DIRECT = """Output only the final answer. Do not show any reasoning, steps, or explanation.
Output exactly one line: "Answer: N" (N is a number only).

Problem: {q}"""

PROMPT_COT = """Solve the following problem step by step.
On the last line, write the final answer as "Answer: N" (N is a number only).

Problem: {q}"""

CONDITIONS = {"A_direct": PROMPT_DIRECT, "B_cot": PROMPT_COT}

# ============================================================
# 3. 모델 호출 — (text, in_tokens, out_tokens, latency_sec) 반환
#    추론 OFF: Claude는 thinking 미사용(기본), Gemini는 thinking_budget=0
#    latency는 '성공한 호출'만 측정 (429 대기 시간 제외)
# ============================================================
from anthropic import Anthropic
from google import genai
from google.genai import types

aclient = Anthropic(api_key=ANTHROPIC_API_KEY)
gclient = genai.Client(api_key=GEMINI_API_KEY)


def call_claude(prompt, model):
    t0 = time.perf_counter()
    r = aclient.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,                      # 재현성
        messages=[{"role": "user", "content": prompt}],
    )
    dt = time.perf_counter() - t0
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens, dt


def call_gemini(prompt, model):
    def make_cfg(thinking_off):
        kw = dict(temperature=0, max_output_tokens=2048)
        if thinking_off:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)  # 추론 OFF
        return types.GenerateContentConfig(**kw)

    cfg = make_cfg(True)
    r, dt = None, 0.0
    for attempt in range(6):
        try:
            t0 = time.perf_counter()
            r = gclient.models.generate_content(model=model, contents=prompt, config=cfg)
            dt = time.perf_counter() - t0          # 성공한 호출만 타이밍
            break
        except Exception as e:
            msg = str(e)
            # 일시적 오류(429 쿼터 + 503/5xx 서버 과부하) → 대기 후 재시도
            transient = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                         "500", "INTERNAL", "502", "504", "overloaded")
            if any(k in msg for k in transient):
                wait = 15 * (attempt + 1)                  # 15, 30, 45 ... 초
                print(f"   ...transient error ({msg[:40]}). waiting {wait}s before retry")
                time.sleep(wait)
            elif cfg.thinking_config is not None:           # thinking 옵션 거부 → 빼고 재시도
                cfg = make_cfg(False)
            else:
                raise
    if r is None:
        return "[retry exceeded]", 0, 0, dt

    text, finish = "", None
    if r.candidates:
        cand = r.candidates[0]
        finish = cand.finish_reason
        if cand.content and cand.content.parts:
            text = "".join(p.text for p in cand.content.parts
                           if getattr(p, "text", None) and not getattr(p, "thought", False))
    if not text:
        text = f"[empty response finish_reason={finish}]"
    u = r.usage_metadata
    return text, (u.prompt_token_count or 0), (u.candidates_token_count or 0), dt

DISPATCH = {"claude" : call_claude, "gemini": call_gemini}

# ============================================================
# 4. 답 추출 — \boxed{} 우선, 다음 "Answer:"/"정답:" 뒤 숫자, 없으면 마지막 숫자
#
#  [v2 수정] v1은 정수만 읽어 두 방향으로 오채점했다.
#   - \boxed{7.00} → 0 으로 추출 (맞은 답을 오답 처리, flash CoT #34)
#   - "Answer: 1.5" → 1 로 절삭 (틀린 답을 정답 처리, Direct #2 3건)
#  이미 저장된 결과는 rescore.py 로 API 재호출 없이 재채점한다.
# ============================================================
NUM = r"-?[\d,]*\.?\d+"


def _to_number(s):
    s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v == int(v) else v


def extract_answer(text):
    boxed = re.findall(r"\\boxed\{\s*\\?\$?\s*(" + NUM + r")\s*\}", text)
    if boxed:
        return _to_number(boxed[-1])
    hits = re.findall(r"(?:정답|answer)\s*[:：]?\s*\$?\s*(" + NUM + r")", text, re.IGNORECASE)
    if hits:
        return _to_number(hits[-1])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return _to_number(nums[-1]) if nums else None

# ============================================================
# 5. 실험 실행 — 4모델 × 2조건 × N문제
# ============================================================
def run():
    rows = []
    total = len(CONDITIONS) * len(MODELS) * len(PROBLEMS)
    n = 0
    for cond_name, tmpl in CONDITIONS.items():
        for model_name in MODEL_ORDER:
            provider, model_str = MODELS[model_name]
            fn = DISPATCH[provider]
            for p in PROBLEMS:
                n += 1
                prompt = tmpl.format(q=p["q"])
                try:
                    text, tin, tout, sec = fn(prompt, model_str)
                    pred = extract_answer(text)
                    correct = (pred == p["gold"])
                except Exception as e:
                    text, tin, tout, sec, pred, correct = f"[ERROR] {e}", 0, 0, 0.0, None, False
                mark = "O" if correct else "X"
                print(f"[{n:3d}/{total}] {cond_name:9s} {model_name:11s} #{p['id']:2d}  "
                      f"pred={str(pred):>8s}  gold={p['gold']:>8d}  {sec:5.1f}s  {mark}")
                rows.append({
                    "model": model_name, "condition": cond_name, "problem_id": p["id"],
                    "question": p["q"], "gold": p["gold"], "pred": pred,
                    "correct": correct, "error_type": "",   # 틀린 것 직접 분류용 빈칸
                    "in_tok": tin, "out_tok": tout, "sec": round(sec, 3),
                    "raw_output": text,
                })
                time.sleep(3)   # 가벼운 rate-limit 안전장치 (필요시 조절)
    return rows

# ============================================================
# 6. 요약 — 4×2 정답률 / 출력토큰 / 평균속도 / 비용 & CSV 저장
# ============================================================
def summarize(rows):
    agg = defaultdict(lambda: {"correct": 0, "n": 0, "tin": 0, "tout": 0, "sec": 0.0})
    for r in rows:
        a = agg[(r["model"], r["condition"])]
        a["n"] += 1
        a["correct"] += int(bool(r["correct"]))
        a["tin"] += r["in_tok"]
        a["tout"] += r["out_tok"]
        a["sec"] += r["sec"]

    print("\n" + "=" * 84)
    print("RESULT SUMMARY (accuracy / out_tokens / avg_latency / cost / cost_per_correct)")
    print("=" * 84)
    header = (f"{'model':12s}{'condition':11s}{'acc':>9s}{'out_tok':>9s}"
              f"{'avg_s':>8s}{'cost($)':>11s}{'$/correct':>11s}")
    print(header)
    print("-" * 84)
    # 모델 순서(작은→큰) × 조건 순서로 정렬해 규모 패턴이 눈에 보이게
    for model_name in MODEL_ORDER:
        for cond_name in CONDITIONS:
            a = agg.get((model_name, cond_name))
            if not a:
                continue
            price = PRICE.get(model_name, {"in": 0, "out": 0})
            cost = a["tin"] / 1e6 * price["in"] + a["tout"] / 1e6 * price["out"]
            per_correct = (cost / a["correct"]) if a["correct"] else float("inf")
            acc = f"{a['correct']}/{a['n']}"
            avg_sec = a["sec"] / a["n"] if a["n"] else 0.0
            pc = "—" if per_correct == float("inf") else f"{per_correct:.6f}"
            print(f"{model_name:12s}{cond_name:11s}{acc:>9s}{a['tout']:>9d}"
                  f"{avg_sec:>8.2f}{cost:>11.6f}{pc:>11s}")

    # CoT 효과(정답률 증분) 한눈에 — Direct→CoT 변화
    print("\n" + "=" * 84)
    print("CoT EFFECT: Direct -> CoT accuracy change (scale emergence / effectiveness)")
    print("=" * 84)
    for model_name in MODEL_ORDER:
        d = agg.get((model_name, "A_direct"))
        c = agg.get((model_name, "B_cot"))
        if d and c:
            d_acc = d["correct"] / d["n"]
            c_acc = c["correct"] / c["n"]
            delta = (c_acc - d_acc) * 100
            print(f"  {model_name:12s} Direct {d_acc*100:5.1f}%  ->  CoT {c_acc*100:5.1f}%   "
                  f"(delta {delta:+5.1f}%p)")

    # 틀린 문제 목록 — '왜 틀렸나' 분석용
    print("\n" + "=" * 84)
    print("WRONG ANSWERS (fill 'error_type' column manually)")
    print("=" * 84)
    for r in rows:
        if not r["correct"]:
            print(f"  {r['model']:12s}{r['condition']:9s} #{r['problem_id']:2d}  "
                  f"pred={r['pred']} / gold={r['gold']}")

    # CSV 저장
    out = "cot_results.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {out}  (raw_output included; open in Excel to classify errors)")


if __name__ == "__main__":
    results = run()
    summarize(results)
