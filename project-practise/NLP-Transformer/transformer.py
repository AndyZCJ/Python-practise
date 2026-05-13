import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim)
        self.position = torch.arange(0, seq_len).unsqueeze(1) #seq, 1
        self.div_term = torch.exp(torch.arange(0, embed_dim, 2)*(-torch.log(10000.0))/embed_dim)
        
        pe[:, 0::2] = torch.sin(self.position * self.div_term)
        pe[:, 1::2] = torch.cos(self.position * self.div_term)
        pe = self.pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        _, seq_len, _ = x.shape
        return x + self.pe[:, :seq_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, embed_dim=512, drop_out=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim//self.num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.drop_out = nn.Dropout(drop_out)
    
    def forward(self, q, k, v, mask=None):
        B, N, D = q.shape
        q, k, v = self.query(q), self.key(k), self.value(v)
        query = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) #B, h, N, d

        attention_score = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.head_dim) # B,h,N,N
        if mask is not None:
            attention_score.masked_fill(mask==0, 1e-9)
        attention = torch.matmul(value, torch.softmax(attention_score, dim=-1)) # B, h, N, d
        attention = attention.transpose(-3, -2).contiguous().reshape(B, N, D)
        return self.drop_out(self.projection(attention))

class FeedForward(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=2048, drop_out=0.1):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        return self.feed_forward(x)

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=2048, num_heads=8,drop_out=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, drop_out)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, drop_out)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_out = nn.Dropout(drop_out)
    
    def froward(self,src, src_mask):
        attention_output = self.attention(src,src,src, src_mask)
        src = self.norm1(src+self.drop_out(attention_output))

        feed_forward_output = self.feed_forward(src)
        src = self.norm2(src + self.drop_out(feed_forward_output))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=2048, num_heads=8,drop_out=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, embed_dim, drop_out)
        self.cross_attention = MultiHeadAttention(num_heads, embed_dim, drop_out)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, drop_out)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop_out = nn.Dropout(drop_out)
    
    def forward(self,x, enc_output, src_mask=None, tgt_mask=None):
        self_attention_output = self.self_attention(x,x,x, tgt_mask)
        x = self.norm1(x+self.drop_out(self_attention_output))
        
        cross_attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.drop_out(cross_attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.drop_out(feed_forward_output))
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, vocab_size, seq_len, embed_dim=512, hidden_dim=2048, num_heads=8,drop_out=0.1):
        super().__init__()
        self.pos_embed = PositionalEmbedding(seq_len, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.ModuleList([
            EncoderLayer(embed_dim, hidden_dim, num_heads,drop_out) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, src, mask=None):
        src = self.embedding(src)
        src = self.pos_embed(src)
        for layers in self.encoder_layer:
            src = layers(src,mask)
        x = self.norm(src)
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, seq_len, embed_dim=512, hidden_dim=2048, num_heads=8,drop_out=0.1):
        super().__init__()
        self.pos_embed = PositionalEmbedding(seq_len, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.ModuleList([
            DecoderLayer(embed_dim, hidden_dim, num_heads,drop_out) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_embed(x)
        for layers in self.decoder_layer:
            x = layers(x, enc_output, src_mask, tgt_mask)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_enc_layers,n_dec_layers, src_vocab_size,tgt_vocab_size, seq_len, embed_dim=512, hidden_dim=2048, num_heads=8,drop_out=0.1):
        super().__init__()
        self.encoder = Encoder(n_enc_layers, src_vocab_size, seq_len, embed_dim, hidden_dim, num_heads,drop_out)
        self.decoder = Decoder(n_dec_layers, tgt_vocab_size, seq_len, embed_dim, hidden_dim, num_heads,drop_out)
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.output_layer(dec_output)
        return output
    
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask==0

    def generate_padding_mask(seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

