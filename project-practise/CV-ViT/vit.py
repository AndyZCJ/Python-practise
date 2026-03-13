import torch
import torch.nn as nn
import math

from mpmath import residual

'''
patch_embed
+ cls
+ pos_embed
↓
N × EncoderBlock
↓
LayerNorm
↓
取 cls token
↓
Linear head

'''

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()

        # TODO 1:
        # num_patches = ?
        self.num_patches = (img_size//patch_size)**2

        # TODO 2:
        # Conv2d: kernel=patch_size, stride=patch_size
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: B,3,32,32

        # TODO 3:
        # conv
        x = self.proj(x) #B, N, 8, 8
        # TODO 4:
        # flatten spatial dims
        x = x.flatten(2)
        # TODO 5:
        # transpose to B,N,D
        x = x.transpose(1, 2)
        return x

class MiniViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128, depth=6, num_heads=4, mlp_ratio=4,
                 num_classes=10, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, mlp_ratio,dropout) for _ in range(depth)])
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #1, 1, D
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim)) #1,N+1,D
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_mlp = nn.Linear(embed_dim, num_classes)

    def forward(self,x):

        # B,N,D
        x = self.patch_embed(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x += self.pos_embed
        x = self.pos_drop(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.final_norm(x) #B,N,D
        cls = x[:,0] #B, D
        logits = self.final_mlp(cls) #B, 10
        return logits

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim//self.num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x) #B,N,3D
        query, key, value = torch.split(qkv,D, dim=-1) #B,N,D

        query = query.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) #B, h, N, d

        attn_score = torch.softmax(torch.matmul(query, key.transpose(-2,-1))/math.sqrt(self.head_dim), dim=-1)
        attn_value = torch.matmul(attn_score, value)

        attn_value = attn_value.transpose(-2, -3).contiguous().reshape(B, N ,D)
        return self.dropout(self.projection(attn_value))

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim,eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim,eps=1e-5)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = residual + self.attention(x)
        residual = x
        x = self.norm2(x)
        x = residual+ self.MLP(x)
        return x

