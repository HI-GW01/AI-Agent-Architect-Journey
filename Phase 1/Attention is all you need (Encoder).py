import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#임베딩 클래스
class Embedding(nn.Module):
    #vocab은 단어 매핑한 딕셔너리, d_model은 차원 수
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        
        # [max_len, d_model] 크기의 PE 행렬 초기화
        pe = torch.zeros(max_len, d_model)
        
        # pos: [max_len, 1]
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        
        # i: [d_model/2] — 짝수 인덱스만 (0, 2, 4, ...)
        i = torch.arange(0, d_model, 2).float()
        
        # 10000^(2i/d_model)
        div_term = torch.pow(10000, i / d_model)
        
        # 짝수 인덱스: sin
        pe[:, 0::2] = torch.sin(pos / div_term)
        # 홀수 인덱스: cos
        pe[:, 1::2] = torch.cos(pos / div_term)
        
        # 학습 파라미터가 아니라 고정값으로 등록
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, d_model]
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V):
        d_k = Q.size(-1)  # 나중에 d_k가 바뀌어도 자동 반영
        
        # Q · Kᵀ / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [seq_len, seq_len]
        
        # 소프트맥스로 확률 변환 (마지막 차원 기준)
        attn_weights = F.softmax(scores, dim=-1)  # [seq_len, seq_len]
        
        # 어텐션 가중치 × V
        output = torch.matmul(attn_weights, V)  # [seq_len, d_model]
        
        return output, attn_weights

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        output, attn_weights = self.scaled_dot_product(Q, K, V)
        return output, attn_weights

# 사용
sentence = "The cat said fuck you"
tokens = sentence.split()   #단어 별로 나누기


vocab = {word: idx for idx, word in enumerate(set(tokens))}
token_ids = torch.tensor([vocab[word] for word in tokens])

embed_layer = Embedding(vocab_size=len(vocab))
pe_layer = PositionalEncoding()

embedded = embed_layer(token_ids)         # [5, 512] 단어별로 임베딩
pe_output = pe_layer(embedded)               # [5, 512] 포지션 임베딩

attention = ScaledDotProductAttention()   # 스케일드 내적 어텐션 객체 선언
output, attn_weights = attention(pe_output)             # 쿼리, 키, 벨류 만듬. 

print("Output shape:", output.shape)  # [5, 512]
print("attn_weights:", attn_weights.shape)  # [5, 5]

