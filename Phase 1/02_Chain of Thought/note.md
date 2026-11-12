# 백지 노트 — CoT 논문 정독

> 논문을 보지 않고(또는 덮고) 백지에 직접 풀어 쓴 정독 메모.
> 다듬지 않은 학습 원본이라, 표현이 거칠어도 그대로 둠.
> 대상 논문: Wei et al., 2022 — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*

---

## Step A — 쌩눈 직관 (이 논문이 뭔 논문인가)

**Abstract (원문)**
> We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

**단어장**
- arithmetic : 산수
- commonsense : 상식

**한 줄 이해**
단지 Chain of Thought를 도입하는 것만으로도 LLM의 산수·상식·상징적 추론 능력이 향상됨.
경험적으로 증명함. (벤치마크: GSM8K = 수학 문제)

**핵심 Figure**
- Figure 1 : 단순한 사고 과정의 예시만 추가했는데 정확도 등이 올라감.
- Figure 2 : PaLM 540B(수학 GSM8K 벤치마크 낮음)에 CoT를 했을 때,
  finetuned된 GPT-3보다 뛰어난 결과(by a large margin). Prior=SOTA를 넘음.

**정리 (내 말로)**
프롬프트 구조만으로 성능(수학에 최적화된 게 아니더라도 수학 능력·추론 능력·문제 해결 능력까지)이
크게 향상된다. 또한 이게 Scale이 큰 모델일수록 파인튜닝 없이 SOTA를 넘는다.

---

## Step B — 더 깊이

**기존과의 차이점**
1. 비용이 많이 드는 finetuning을 배제
2. 기존 퓨샷(질문·정답 쌍)이 복잡한 추론·수학·상식 문제에서 성능 저조하다는 한계 극복
3. 질문과 정답을 연결하는 자연어 추론 과정을 프롬프트에 담는 새로운 방법론

**CoT의 장점**
1. 컴퓨팅 자원을 유연하게 할당 (문제 난이도에 따라)
2. 모델이 답에 어떻게 도달했는지 보여주어 해석 가능성 높임 (Explainable AI), 디버깅 가능
3. 다양한 문제에 범용적 적용 가능
4. 기성 모델에 손쉬운 적용

**왜 CoT가 작동하는가 (ablation)**
논문은 세 가지 변형으로 검증:
- equation only (수식만)
- variable compute only (계산량만 늘림)
- reasoning after answer (답 먼저, 추론 나중)
→ 셋 다 효과 없었음.
즉 "단순히 중간 계산 토큰을 늘려서"가 아니라,
**자연어로 추론 과정을 표현하는 것 자체가 핵심**이라는 증거.

**Emergent Ability (창발)**
100B 이하 모델에서는 CoT가 오히려 성능을 깎는다.
→ "CoT는 아무 모델에나 쓰는 기법이 아니라, 모델 규모가 전제조건"이라는 실용적 판단 기준.

---

## 정독 후 떠오른 의문 → 실험으로 이어짐

> 2022년 논문은 "100B 미만은 CoT 효과 약함"이라고 했다.
> 그런데 2025년의 Gemini Flash-Lite 같은 모델은 **큰 모델에서 증류(distill)된 소형 모델**이다.
> teacher의 추론 패턴을 물려받았다면, 크기가 작아도 CoT 효과가 살아있지 않을까?
> → "100B 창발" 통념이 증류 모델에서는 깨질 것이다 = 이번 실험의 핵심 가설.

(실험 설계·결과는 [`README.md`](README.md), 코드는 [`CoT_test.py`](CoT_test.py))
