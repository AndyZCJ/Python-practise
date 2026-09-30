'''
debug配置:6 layers, hidden size 384, 6 heads, context 256
full 配置: 12 layers, hidden size 768, 12 heads, context 512
'''
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
@dataclass
class config:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float
    layer_norm_eps: float
    use_bias: bool

    def __post_init__(self):
        if self.embedding_dim%self.num_heads!=0:
            raise ValueError("embedding_dim必须能被num_heads整除")
        
        if self.embedding_dim*4 != self.ffn_dim:
            raise ValueError("ffn_dim必须是embedding_dim的四倍")


debug_config = config(vocab_size=16384, context_length=256, embedding_dim=384, num_heads=6,
                      num_layers=6, ffn_dim=1536, dropout=0.1, layer_norm_eps=1e-5, use_bias=True)        
full_config = config(vocab_size=16384, context_length=512, embedding_dim=768, num_heads=12,
                      num_layers=12, ffn_dim=3072, dropout=0.1, layer_norm_eps=1e-5, use_bias=True)

class GPT_Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, dropout):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(self.context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        _, L = x.shape
        if L > self.context_length:
            raise ValueError("输入序列长度不能超过context_length")
        
        ids = torch.arange(0, L).to(x.device)
        pos_embed = self.pos_embedding(ids)
        token_embed = self.token_embedding(x)
        x = pos_embed + token_embed
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, use_bias, dropout):
        super().__init__()
        if embedding_dim%num_heads != 0:
            raise ValueError("embedding_dim必须能被num_heads整除")
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = embedding_dim//self.num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.register_buffer("mask",
                         torch.triu(torch.ones(self.context_length, self.context_length,dtype=torch.bool).unsqueeze(0).unsqueeze(0),
                                    diagonal=1))
        

    def forward(self, x):
        B, L, D = x.shape
        if L>self.context_length:
            raise ValueError("L不能大于最大序列长度context_length")
        mask = self.mask[:, :, :L, :L]
        query = self.query(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_score = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.head_dim)
        attention_score = attention_score.masked_fill(mask=mask, value=float('-inf'))
        attention_score = torch.softmax(attention_score, dim=-1)
        attention_score = self.attn_dropout(attention_score)
        attention_value = torch.matmul(attention_score, value).permute(0, 2, 1, 3).contiguous().reshape(B, L, D)
        attention_value = self.projection(attention_value)
        return self.proj_dropout(attention_value)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, use_bias, ffn_dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim, bias=use_bias),
            nn.Dropout(ffn_dropout)
        )

    def forward(self, x):
        return self.ffn(x)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, num_heads, context_length, use_bias, layer_norm_eps,dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, context_length, use_bias, dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps)
        self.ffn = FeedForward(embedding_dim, ffn_dim, use_bias, dropout)
        
    def forward(self, x):
        tmp = x
        x = self.norm1(x)
        x = self.attention(x)
        x = tmp + x

        tmp = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = tmp + x
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = GPT_Embedding(config.vocab_size, config.embedding_dim, config.context_length, config.dropout)
        self.transformer_blocks = nn.ModuleList(TransformerBlock(config.embedding_dim, config.ffn_dim, config.num_heads, 
                                                                  config.context_length, config.use_bias, config.layer_norm_eps,config.dropout)
                                                                  for i in range(config.num_layers))
        self.norm = nn.LayerNorm(normalized_shape=config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embedding.token_embedding.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02,)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x) #权重共享

def gpt_loss(logits, target):
    B, L, vocab_size = logits.shape
    loss = F.cross_entropy(logits.reshape(B*L, vocab_size), target.reshape(B*L))
    return loss