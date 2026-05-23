# AI-Agent-Architect-Journey
AI-Agent 마스터하기

## Step 1: Attention is All You Need (Transformer)

### 📄 논문 개요 (Abstract Review)
- **핵심 아이디어:** 기존의 복잡한 RNN, CNN 구조를 완전히 제거하고 오직 **Attention Mechanism**만으로 네트워크를 설계 (Transformer).
- **성과:** 번역 품질(BLEU 스코어)은 높이면서, 병렬 처리가 가능해져 학습 시간은 획기적으로 단축함.
- **PM적 시각:** '순차적 처리'라는 시스템의 병목을 제거하여 생산성을 극대화한 혁신적인 아키텍처.

### 💡 산업공학도의 인사이트
- **공정의 병렬화:** RNN이 '직렬 공정'이라면 Transformer는 '병렬 공정'이다. 자원을 동시에 투입할 수 있다는 것은 곧 서비스 확장성(Scalability)과 직결된다.
- **Attention의 본질:** 수많은 데이터 중 '어디에 집중할 것인가'를 스스로 결정하는 가중치 시스템. 이는 에이전트가 사용자 요청의 핵심 의도를 파악하는 논리적 근간이 된다.

### ❓ 추가 학습 키워드
- [ ] RNN의 기울기 소실(Vanishing Gradient) 문제
- [ ] Self-Attention의 수학적 원리 (Q, K, V)
- [ ] BLEU Score 외에 최신 LLM 평가 지표들