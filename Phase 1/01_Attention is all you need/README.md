01 · Attention Is All You Need


RNN의 순차 병목과 CNN의 깊은 층 의존을 버리고, 어텐션만으로 모든 토큰의 관계를 병렬 계산하는 Transformer.
이 폴더는 논문을 읽고 끝낸 게 아니라, PyTorch로 밑바닥부터 직접 구현하고 forward를 검증한 기록이다.
Phase 1 · 구현 깊이 L5 (로드맵에서 유일한 밑바닥 구현 논문).




📄 논문 한 줄 (Logic)


기존 문제: RNN은 토큰을 한 개씩 직렬 처리해 병목이 생기고, CNN은 멀리 떨어진 단어를 잇는 데 층이 많이 필요하다.
제안: Self-Attention만으로 모든 토큰의 관계를 한 번에(병렬) 계산하는 인코더–디코더 아키텍처.
결과: 번역 품질(BLEU) 향상 + 병렬화로 학습 시간 대폭 단축.



🏭 PM·산업공학 인사이트 (Insight)


공정의 병렬화 — RNN이 '한 줄짜리 컨베이어 벨트(직렬)'라면, Transformer는 '다단계 병렬 가공 라인'이다. 자원을 동시에 투입할 수 있어 대규모 학습이 가능해진 지점.
학습은 병렬, 추론은 순차 (가장 중요한 한 줄) — 학습 때는 정답을 통째로 넣고 마스킹으로 한 번에 처리하지만, 생성(추론)은 토큰을 하나씩 만들어 다시 입력으로 넣는 자기회귀라 순차적이다. → LLM이 왜 느리고(지연), 토큰당 과금이며(비용), 스트리밍 UX가 필요한지의 근원이 바로 이 비대칭이다.
Attention vs Feed Forward — Attention이 단어들 사이의 *관계(Network)*를 조율한다면, Feed Forward는 각 단어 고유의 *특징(Feature)*을 더 깊게 가공하는 독립 공정이다.
Add & Norm = 품질 가드레일 — 레이어를 거치며 원본 정보가 유실되지 않게 더해주고(residual), 더한 뒤 분포가 튀지 않게 표준화(norm)한다.



🧱 구현 (Code)

밑바닥부터 PyTorch로 작성. 검증 원칙은 "내가 말로 설명 → AI는 코드만 생성, 에러는 힌트만".
👉 src/transformer.py

서브블록역할Embedding토큰 인덱스 → 512차원 벡터PositionalEncodingsin/cos로 순서 정보 주입 (학습 X, 고정값)ScaledDotProductAttentionQ·Kᵀ / √d_k → softmax → V 가중합MultiHeadAttention512를 8헤드(64차원)로 쪼개 병렬 어텐션 후 concatFeedForward512 → 2048 → 512, ReLU 비선형EncoderLayer / DecoderLayer(Masked) MHA · Cross-MHA · FFN + 각각 Add & NormOutputLayer512 → 단어장 크기 Linear → softmaxTransformer위를 묶어 src → 예측 확률분포까지

forward 검증: src 7토큰 / tgt 8토큰을 넣어 [8, tgt_vocab] 확률분포가 끝까지 흐르는 것 확인.
(학습 루프·배치 처리는 의도적으로 범위 밖 — 목표는 '구조의 이해'이지 번역기 학습이 아님.)


🐛 직접 부딪힌 지점 (실패 → 진단 → 수정)

포트폴리오의 핵심은 여기다. 책으로 읽는 것과, 직접 짜다 막혀서 추론으로 푸는 것은 다르다.


증상 — src(7토큰)·tgt(8토큰)로 돌리자 cross-attention에서 터짐:
RuntimeError: shape '[8, 8, 64]' is invalid for input of size 3584
원인 — MultiHeadAttention이 seq_len = query.size(0) 하나로 Q·K·V를 전부 reshape했다. self-attention에선 Q=K=V라 길이가 같아 우연히 통과했지만, cross-attention은 Q=디코더(8토큰), K·V=인코더(7토큰) 로 길이가 다르다.
핵심 깨달음 — reshape/view는 데이터를 늘리거나 줄이지 않는다. 원소 개수를 보존한 채 모양만 바꾼다. 7토큰(3584개)을 8토큰 모양(4096개)으로 우겨넣는 건 애초에 불가능.
수정 — 길이를 q_len / k_len / v_len으로 분리해 각 텐서가 자기 길이로 reshape하도록 변경. (어텐션은 Q와 K·V의 개수가 달라도 된다 — 점수 행렬이 [q_len, k_len]이라 독립이니까.)



추론 생성 루프에선 디코더 입력이 1 → 2 → 3 …으로 계속 길어지므로, Q와 K·V의 길이가 다른 건 예외가 아니라 기본 상황. 이 수정은 선택이 아니라 필수였다.




✅ 해결한 의문 (이전 To-be-Solved)


왜 √d_k로 스케일링하나? — d_k가 커지면 내적값이 커지고, softmax가 기울기 ≈ 0인 영역으로 밀려 학습이 죽는다. 그걸 막기 위한 표준화.
순차 데이터가 아닌데 위치를 어떻게 기억하나? — sin/cos 주기 함수로 각 위치에 고유한 '순번표'를 만들어 임베딩에 더한다. 상대 위치가 선형 관계로 표현돼 길이 일반화에도 유리.



📊 학습 키워드


BLEU Score — 기계 번역이 사람 번역과 얼마나 유사한지 수치화(0~100).
Add & Norm — residual(정보 보존) + LayerNorm(분포 안정화).
Auto-regressive — 자기 출력을 다음 입력으로 다시 쓰는 순차 생성.



🛠️ 진행 현황

 Abstract · Figure 1 구조 분석
 Scaled Dot-Product 스케일링(√d_k) 이유 이해
 Positional Encoding 직관 이해
 전체 Transformer 밑바닥 구현 + forward 검증
 cross-attention 길이 버그 진단 · 수정
 백지 자연어 재현 (L5 통과 기준) → notes.md