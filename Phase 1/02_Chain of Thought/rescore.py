# -*- coding: utf-8 -*-
"""
CoT 실험 재채점 스크립트 (API 재호출 없음)

cot_results.csv에 저장된 raw_output만 다시 채점한다.
오답을 전수 확인하며 찾은 평가 쪽 오류(라벨·문항·채점기)를 바로잡고,
원시 채점과 재채점 결과를 나란히 보여 준다.

오류 카탈로그
  E-01 라벨 오류   #1   정답 라벨 2280, 실제 정답 2180 (430+750+300+700)
  E-02 문항 오류   #45  문장에 "ten stalls"와 "twenty stalls"가 함께 나온다.
                        10칸 해석이면 192, 20칸 해석이면 176. 두 답을 모두 정답으로 인정한다.
  E-03 채점기 버그  v1 정규식이 정수만 읽었다.
                        - #34 flash CoT: \\boxed{7.00} 을 0으로 추출해 맞은 답을 오답 처리
                        - #2 Direct: 1.5를 1로 잘라 읽어 틀린 답 3건을 정답 처리
  (참고) Direct 추론 누출: "답만 출력" 지시에도 풀이를 함께 낸 응답(출력 30토큰 이상)은
         Direct 집계에서 분리한다. 실제로는 sonnet 6건만 해당한다.

실행
  python rescore.py
  → 요약표 출력 + cot_results_rescored.csv 저장
"""

import csv
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cot_results.csv")
OUT = os.path.join(HERE, "cot_results_rescored.csv")

MODEL_ORDER = ["flash-lite", "haiku", "flash", "sonnet"]
LEAK_TOKENS = 30  # Direct 정상 응답은 7~9토큰. 이 이상이면 풀이가 섞인 것으로 본다.

# 문항별 정답 정정: problem_id -> (인정하는 정답 집합, 오류 유형)
CORRECTIONS = {
    "1": ({2180}, "label_error"),
    "45": ({176, 192}, "item_error"),
}


# ------------------------------------------------------------
# v1 채점기 (CoT_test.py 원래 버전): 정수만 읽음
# ------------------------------------------------------------
def extract_answer_v1(text):
    hits = re.findall(r"(?:정답|answer)\s*[:：]?\s*(-?[\d,]+)", text, re.IGNORECASE)
    if hits:
        try:
            return int(hits[-1].replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"-?\d[\d,]*", text)
    if nums:
        try:
            return int(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


# ------------------------------------------------------------
# v2 채점기: \boxed{} 우선, 소수까지 읽음
# ------------------------------------------------------------
NUM = r"-?[\d,]*\.?\d+"


def _to_number(s):
    s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v == int(v) else v


def extract_answer_v2(text):
    boxed = re.findall(r"\\boxed\{\s*\\?\$?\s*(" + NUM + r")\s*\}", text)
    if boxed:
        return _to_number(boxed[-1])
    hits = re.findall(r"(?:정답|answer)\s*[:：]?\s*\$?\s*(" + NUM + r")", text, re.IGNORECASE)
    if hits:
        return _to_number(hits[-1])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return _to_number(nums[-1]) if nums else None


def main():
    with open(SRC, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        pid, gold = r["problem_id"], int(r["gold"])
        pred_v2 = extract_answer_v2(r["raw_output"])
        accepted, label_type = CORRECTIONS.get(pid, ({gold}, ""))
        correct_v1 = r["correct"].strip().upper() == "TRUE"
        correct_v2 = pred_v2 in accepted
        leak = r["condition"] == "A_direct" and int(r["out_tok"]) >= LEAK_TOKENS

        error_type = ""
        if correct_v1 != correct_v2:
            if label_type:
                error_type = label_type
            elif correct_v2:
                error_type = "parse_false_negative"
            else:
                error_type = "parse_false_positive"
        if leak:
            error_type = (error_type + ";" if error_type else "") + "reasoning_leak"

        out_rows.append({
            "model": r["model"],
            "condition": r["condition"],
            "problem_id": pid,
            "gold": gold,
            "accepted_answers": "|".join(str(a) for a in sorted(accepted)),
            "pred_v1": r["pred"],
            "pred_v2": pred_v2,
            "correct_v1": correct_v1,
            "correct_v2": correct_v2,
            "error_type": error_type,
            "excluded_leak": leak,
            "out_tok": r["out_tok"],
        })

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # 요약: Direct는 추론 누출 응답을 분리한 기준
    acc = defaultdict(lambda: {"n": 0, "v1": 0, "v2": 0})
    for r in out_rows:
        if r["excluded_leak"]:
            continue
        a = acc[(r["model"], r["condition"])]
        a["n"] += 1
        a["v1"] += r["correct_v1"]
        a["v2"] += r["correct_v2"]

    def pct(model, cond, key):
        a = acc[(model, cond)]
        return a[key] / a["n"] * 100, a["n"]

    print("모델        | Direct v1→v2 (n)        | CoT v1→v2 (n)          | Δ v1→v2")
    print("-" * 78)
    for m in MODEL_ORDER:
        d1, dn = pct(m, "A_direct", "v1")
        d2, _ = pct(m, "A_direct", "v2")
        c1, cn = pct(m, "B_cot", "v1")
        c2, _ = pct(m, "B_cot", "v2")
        print(f"{m:11s} | {d1:5.1f}% → {d2:5.1f}% ({dn})   | {c1:5.1f}% → {c2:5.1f}% ({cn})  "
              f"| {c1 - d1:+5.1f} → {c2 - d2:+5.1f}%p")

    changed = [r for r in out_rows if r["correct_v1"] != r["correct_v2"]]
    print(f"\n판정이 바뀐 응답 {len(changed)}건")
    for r in changed:
        print(f"  {r['model']:11s} {r['condition']:9s} #{r['problem_id']:>2s}  "
              f"pred {r['pred_v1']} → {r['pred_v2']}  "
              f"{'O' if r['correct_v1'] else 'X'} → {'O' if r['correct_v2'] else 'X'}  ({r['error_type']})")
    leaks = [r for r in out_rows if r["excluded_leak"]]
    print(f"\nDirect 추론 누출로 분리한 응답 {len(leaks)}건: "
          + ", ".join(f"{r['model']} #{r['problem_id']}" for r in leaks))
    print(f"\nSaved -> {os.path.basename(OUT)}")


if __name__ == "__main__":
    sys.exit(main())
