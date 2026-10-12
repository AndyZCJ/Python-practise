import torch
import torch.nn as nn
def window_partition(x, window_size):
    """
    x: (B, C, H, W)
    return:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, C, H, W = x.size()
    num_windows = (H//window_size) * (W//window_size)
    x = x.view(B, C, H//window_size, window_size, W//window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.reshape(B*num_windows, window_size*window_size, C)
    return x

def window_reverse(windows, window_size, H, W):
    """
    windows: (B*num_windows, w*w, C)
    return: (B, C, H, W)
    """
    # Step 1: 推回 (B, H//w, W//w, w, w, C)
    # Step 2: permute 回 (B, C, H//w, w, W//w, w)
    # Step 3: reshape 回 (B, C, H, W)

    num_windows = (H//window_size) * (W//window_size)
    assert windows.shape[0] % num_windows==0
    B = windows.shape[0] // num_windows
    C = windows.shape[-1]
    windows = windows.reshape(B, H//window_size,W//window_size,window_size,window_size, C)
    windows = windows.permute(0, 5, 1, 3, 2, 4).contiguous()
    return windows.reshape(B, C, H, W)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # w
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size-1)*(2 * self.window_size-1), self.num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij')) # (2, w, w)
        coords_flatten = coords.flatten(1) #(2, N) N=w*w
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() #(N, N, 2)

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size -1
        relative_position_index = relative_coords.sum(-1) #(N, N)

        self.register_buffer("relative_position_index", relative_position_index)
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 下一步再加：relative_position_bias_table / index
        # 这一步先做“无相对位置偏置”的版本

    def forward(self, x, mask=None):
        """
        x: (B*nW, N, C) where N = w*w
        mask:(nW, N, N)
        return: (B*nW, N, C)
        """
        # 1) qkv = self.qkv(x)
        # 2) reshape/split to q,k,v with heads
        # 3) attn = (q @ k^T) * scale
        # 4) softmax -> dropout
        # 5) out = attn @ v -> merge heads
        # 6) proj -> dropout

        B_, N, C = x.shape
        qkv = self.qkv(x)
        query, key, value = torch.split(qkv, C, dim=-1)
        query = query.reshape(B_, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key.transpose(-2, -1))*self.scale# (B_, heads, N, N)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, self.num_heads)# (N, N, heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # (heads, N, N)
        attention = attention + relative_position_bias.unsqueeze(0)
        if mask is not None:
            mask = mask.to(attention.dtype)
            num_windows = mask.shape[0]
            attention = attention.reshape(B_//num_windows, num_windows, self.num_heads, N, N)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(B_, self.num_heads, N, N)
        attention_score = self.attn_drop(torch.softmax(attention, dim=-1))
        attention_value = torch.matmul(attention_score, value)
        attention_value = attention_value.transpose(1, 2).contiguous().reshape(B_, N, C)
        attention_value = self.proj_drop(self.proj(attention_value))
        return attention_value

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        H, W = input_resolution
        self.H, self.W = H, W
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """
        x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == self.H * self.W

        shortcut = x
        x = self.norm1(x)
        device = x.device
        assert self.H % self.window_size==0 and self.W%self.window_size==0
        mask = None
        if self.shift_size>0:
            mask = self.create_attn_mask(self.H, self.W, device)
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(B, C, self.H, self.W) #(B, C, H, W)
        if self.shift_size>0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(2,3))
        x = window_partition(x, self.window_size) #(B*nW, w*w, C)
        x = self.attn(x, mask)
        x = window_reverse(x, self.window_size, self.H, self.W) #(B, C, H, W)
        if self.shift_size>0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(2, 3))
        x = x.reshape(B, C, L).permute(0, 2, 1).contiguous() #(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    
    def create_attn_mask(self, H, W, device):
        """
        return: mask, shape = (nW, N, N), N = window_size * window_size
        """
        img_mask = torch.zeros((1, H, W, 1), device=device)

        w = self.window_size
        s = self.shift_size

        h_slices = (slice(0, -w), slice(-w, -s), slice(-s, None))
        w_slices = (slice(0, -w), slice(-w, -s), slice(-s, None))

        cnt = 0
        for h in h_slices:
            for w_ in w_slices:
                img_mask[:, h, w_, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask.permute(0, 3, 1, 2), self.window_size)# (nW, N, 1)
        mask_windows = mask_windows.squeeze(-1) # (nW, N)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask
