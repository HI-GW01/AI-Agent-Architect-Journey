# Chain-of-Thought 재현 실험 (GSM8K)

> Wei et al. (2022)의 Chain-of-Thought 주장을, 2025년 상용 소형 모델 4종에서 직접 재현하고 확장한 실험.
> 논문을 읽고 끝내지 않고, 직접 코드를 짜서 돌려보고 숫자로 검증한 기록입니다.

---

## 1. 무엇을 / 왜 검증했나

[Wei et al., 2022 — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903)의
핵심 주장은 세 가지입니다.

1. **프롬프트 구조만으로** 파인튜닝 없이 추론 성능이 크게 오른다.
2. 이 효과는 **모델 규모가 클수록(~100B+)** 창발(emergent)한다.
3. 100B 이하 소형 모델에서는 CoT가 오히려 성능을 깎을 수 있다.

논문은 2022년 PaLM·GPT-3 기준입니다. 그래서 두 가지가 궁금했습니다.

- **(Q1) 재현성** — 2025년 최신 상용 모델에도 zero-shot CoT가 여전히 효과적인가?
- **(Q2) 창발 가설 검증** — 큰 teacher에서 **증류(distill)된 소형 모델**(Gemini Flash-Lite)은
  "100B 창발" 통념을 깨고 소형임에도 CoT 효과를 보일 것이다 — 라는 가설을 직접 세워 검증.

> 원 논문은 few-shot CoT(예시 제공)지만, 이 실험은 **zero-shot CoT**("step by step" 지시만)로
> 단순화했습니다. 조건 폭발을 막고 "프롬프트가 유도한 추론"만 변수로 격리하기 위함입니다.
> (논문 정독 메모는 [`note.md`](note.md) 참고)

---

## 2. 실험 설계

**4 모델 × 2 조건 × 50 문제 = 400 API calls**

| 축 | 값 |
|---|---|
| 모델 | `haiku`(Claude Haiku 4.5), `sonnet`(Claude Sonnet 4.6), `flash-lite`(Gemini 2.5 Flash-Lite), `flash`(Gemini 2.5 Flash) |
| 조건 A — Direct | 풀이 없이 답만 한 줄로 출력 |
| 조건 B — CoT | "step by step"으로 풀고 마지막 줄에 답 |
| 데이터 | [GSM8K](https://github.com/openai/grade-school-math) test set에서 seed 고정 무작위 50문제 |
| 측정 | 정답률 / 출력 토큰 / 지연 / **정답당 비용** |

**변수 격리를 위한 통제**
- `temperature=0` (재현성)
- **추론 기능 전 모델 OFF** (Claude thinking 미사용, Gemini `thinking_budget=0`)
  → 모델 내장 추론이 아니라 **프롬프트가 유도한 CoT만** 변수로 남김
- 두 조건의 출력 형식(`Answer: N`) 통일 → 채점 파서 일관성

---

## 3. 결과

`N=50`, 추론 OFF, `temperature=0`. (raw 데이터: [`cot_results.csv`](cot_results.csv))

### CoT 효과: Direct → CoT 정답률 변화

| 모델 | Direct | CoT | Δ |
|---|---:|---:|---:|
| **flash-lite** (증류 소형) | 36.0% | 96.0% | **+60.0%p** |
| **haiku** (소형) | 44.0% | 98.0% | **+54.0%p** |
| **flash** (중형) | 56.0% | 94.0% | +38.0%p |
| **sonnet** (대형) | 68.2%¹ | 96.0% | +27.8%p |

¹ Sonnet Direct는 추론 누출 6건(발견 ④)을 제외한 44문항 기준(30/44). 누출 분리 전 원시값은 72.0%(36/50).

### 비용·속도

| 모델 | 조건 | 정답률 | 출력토큰(합) | 평균 지연 | 비용($) |
|---|---|---:|---:|---:|---:|
| haiku | Direct | 22/50 | 369 | 1.08s | 0.00725 |
| haiku | CoT | 49/50 | 9,351 | 2.03s | 0.05201 |
| flash-lite | Direct | 18/50 | 277 | 1.97s | 0.00060 |
| flash-lite | CoT | 48/50 | 14,014 | 3.25s | 0.00607 |
| sonnet | Direct¹ | 30/44 | 318 | 1.86s | 0.01840 |
| sonnet | CoT | 48/50 | 7,462 | 3.42s | 0.12769 |
| flash | Direct | 28/50 | 275 | 1.08s | 0.00215 |
| flash | CoT | 47/50 | 12,223 | 2.01s | 0.03196 |

¹ Sonnet Direct만 추론 누출 6건을 분리한 44문항 기준(발견 ④). 다른 셀은 모두 50문항 원시값.

---

## 4. 핵심 발견

**① CoT는 2025년 상용 모델에서도 여전히 강력하다 (Q1 ✅)**
4개 모델 전부 정답률이 올랐고, 모두 CoT에서 94~98%로 수렴.
프롬프트 한 줄 차이가 정답률을 27.8~60%p 끌어올렸습니다.

**② "100B 창발" 통념은 증류 소형 모델에서 깨진다 (Q2 ✅)**
가장 작은 축인 **Flash-Lite가 +60.0%p로 CoT 증분 1위**.
논문(2022)대로라면 소형 모델은 CoT 효과가 약해야 하지만,
큰 teacher에서 증류된 2025년 소형 모델은 작아도 추론 패턴을 물려받아 CoT를 강하게 활용했습니다.
→ **창발은 "절대 크기"가 아니라 "학습 방식"의 함수**라는 해석.

**③ 비용 효율은 Flash-Lite의 압승**
CoT 비용 Flash-Lite $0.00607 vs Sonnet $0.12769 — 약 **21배** 차이.
정답률은 96% vs 96%로 동일. "정확도당 비용"에선 소형 증류 모델이 정답.

**④ [방법론적 발견] Sonnet의 Direct 조건 추론 누출**
"답만 출력하라"고 **똑같이** 지시했는데 **4개 모델 중 Sonnet만** 일부 문항에서 풀이를 함께 출력
(누출 6건, 출력 토큰이 7토큰대→최대 89토큰까지 튐).
그대로 두면 Sonnet Direct의 토큰·비용·정답률이 왜곡되므로, 이 6건을 분리해
**44문항 기준으로 재집계**했습니다(정답률 72.0% → 68.2%, CoT 델타 +27.8%p).
결과를 곧이곧대로 믿지 않고 이상치를 골라내 분리한 지점입니다.

---

## 5. 재현 방법

```bash
pip install anthropic google-genai
# cot_experiment.py 상단 ANTHROPIC_API_KEY, GEMINI_API_KEY 채우기
python cot_experiment.py
```
GSM8K test set은 최초 1회 자동 다운로드. 결과는 `cot_results.csv`로 저장(raw_output 포함).
스모크 테스트는 `MODEL_ORDER`를 1개로 줄이고 `N_PROBLEMS=5`로 먼저 파이프라인 확인 권장.

---

## 6. 한계

- N=50 단일 시행 — 신뢰구간 확보엔 반복·문항 확대 필요
- zero-shot CoT만 검증 (원 논문의 few-shot은 범위 밖)
- 모델 4종은 "규모 그라데이션"의 근사일 뿐, 정확한 파라미터 수는 비공개
- 정답 추출이 정규식 기반 → 드문 포맷 이탈은 오답 처리될 수 있음

---

*self-directed AI 학습 로드맵의 일부 (CoT = L1 개념 이해 단계). 설계·가설은 직접, 학습은 NotebookLM, 구현은 Claude, 실행은 Gemini/Claude API.*