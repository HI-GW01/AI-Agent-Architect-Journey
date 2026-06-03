# Attention is All You Need

## 기존과 차이점
- RNN 순차적이라는 특성, CNN은 층이 너무 많다는 특성 때문에 병목이 생김.
- 기존에 Attention을 사용하기는 했지만 단독으로는 없었음.
- Attention만으로 생성한 아키텍쳐는 모든 토큰을 병렬적으로 계산 가능.

## 구조
- 인코더와 디코더로 구성됨.
- 인코더, 디코더 모두 6개의 Layer가 존재. (각 layer의 구조는 동일하나 가중치가 다르게 6번 반복함으로써 해당 문장을 더 깊게 이해 & 출력)

### 인코더
- Embedding : 문장을 토큰화(벡터화, 512차원으로) (N개의 단어가 있는 문장을 인풋하면 `[N, 512]` 크기를 가짐)
- Positional Encoding : 문장내 순서, 벡터의 순서를 활용하여 sin, cos 함수를 각 벡터값에 순서 벡터를 더해 각 단어(=토큰 단위로 나눠진 벡터값들)에 순서를 매김

#### 인코더의 핵심 레이어
- 전체 구조 : Multi-head Attention -> Add&Norm -> Feed-Forwarding -> Add&Norm
- Multi-head Attention : Scaled-dotproduction-attention을 진행함. 단 Q, K, V를 8등분함. 가중치가 다른 8개(이 논문에서 도출한 최적값, 고정 아님)의 Multi-head로 진행한 뒤 마지막에 병합(Concate)
- Scaled-dotproduction-attention : 모든 토큰을 Q, K, V로 만든 뒤 해당 Q와 모든 K의 관련한 정도(Q와 K 벡터간의 내적)를 도출함. Q, K 내적을 root(64(512/8))으로 나눠 스케일링한 뒤 그 Softmax 함수를 통해 확률화(한 Q에 대해 모든 K의 softmax 값을 합하면 1이 됨.) 거기에 V를 곱함으로서 마무리.
- Add&Norm : 현재 통과 중인 layer의 출력과 원본을 합침. (Add; ResNet에서 제안되며 겹겹의 레이어 사이에서 원본데이터 유실을 방지) 그냥 더하면 분포가 너무 커져버리니 정규화(Normalization)
- Feedforwarding : 512차원을 4배로 확장. (`[N, 512]`에 `[512, 2048]`을 곱하여 `[N, 2048]`) 그 뒤 ReLU 활성화 함수를 사용해 음수는 모두 0으로 날려버림. (ReLU가 없었다면 `W_1`, `W_2`와 관련된 두 선형의 합성(행렬곱)으로 선형적이기만 했을 거임. 그러면 Precise한 예측 불가능, 음수를 0으로 날려버림으로서 비선형, 그것도 아주 복잡한 2048에서 중간중간 0으로 극단적인 꺾임을 구현) 다시 512차원으로 복귀(`[2048, 512]` 곱하기)

참고로 인코더, 디코더 모두 피드포워딩이 활용됨. 정확한 이름은 Position-wise Feed-Forward Networks.

이렇게 구성된 인코더의 sublayer를 6번 반복 나온 결과가 encoder에 전달됨.