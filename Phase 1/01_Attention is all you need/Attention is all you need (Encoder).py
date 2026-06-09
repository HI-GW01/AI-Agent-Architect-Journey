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

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)

        # 마스크가 있으면 막을 위치를 -무한대로 (softmax 통과하면 0이 됨)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)  
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.h = h
        self.d_k = d_model // h  # 512 // 8 = 64

        # 작은 행렬 h개를 쌓은 것과 동일한 큰 행렬 하나씩
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # concat 후 최종 투영

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=False):
        seq_len = query.size(0)

        # 1) 투영: [seq, 512]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2) 헤드로 split: [seq, 512] -> [seq, h, d_k] -> [h, seq, d_k]
        Q = Q.view(seq_len, self.h, self.d_k).permute(1, 0, 2)
        K = K.view(seq_len, self.h, self.d_k).permute(1, 0, 2)
        V = V.view(seq_len, self.h, self.d_k).permute(1, 0, 2)

        # 3) 마스크 체크 시: 미래 위치(상삼각)를 막는 마스크 생성
        causal_mask = None
        if mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # 4) attention: [h, seq, d_k]
        out, attn_weights = self.attention(Q, K, V, causal_mask)

        # 5) concat: [h, seq, d_k] -> [seq, h, d_k] -> [seq, 512]
        out = out.permute(1, 0, 2).reshape(seq_len, self.h * self.d_k)

        # 6) 최종 투영 W_o
        out = self.W_o(out)
        return out, attn_weights
    
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: [seq_len, d_model]
        x = self.W1(x)           # [seq_len, 2048]
        x = F.relu(x)            # 음수 제거 (비선형)
        x = self.W2(x)           # [seq_len, 512]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=False):
        # 1) Multi-Head Attention -> Add & Norm
        attn_out, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + attn_out)

        # 2) FeedForward -> Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, h=8, d_ff=2048, N=6):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for _ in range(N)])

    def forward(self, x, mask=False):
        x = self.embedding(x)           # 토큰 -> 벡터
        x = self.positional_encoding(x) # 위치 정보 추가
        for layer in self.layers:       # EncoderLayer 6번 순차 통과
            x = layer(x, mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, h)
        self.cross_mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=True):
        # 1) Masked MHA -> Add & Norm
        attn_out, _ = self.masked_mha(x, x, x, mask)
        x = self.norm1(x + attn_out)

        # 2) Cross MHA (Q: 디코더, K/V: 인코더) -> Add & Norm
        attn_out, _ = self.cross_mha(x, encoder_output, encoder_output, mask=False)
        x = self.norm2(x + attn_out)

        # 3) FeedForward -> Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, h=8, d_ff=2048, N=6):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff) for _ in range(N)])

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x
    
# 사용
sentence = "The cat said we can do everything"
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

