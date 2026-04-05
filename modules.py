import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
        # t: [max_seq_len]
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # freqs: [max_seq_len, dim / 2]
        freqs = torch.outer(t, inv_freq)
        
        # We need to duplicate each freq to match the [..., dim] shape for rotary product
        # i.e. (freq_1, freq_1, freq_2, freq_2, ...)
        
        # Register as buffers so they're moved to the correct device but not updated by optimizer
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, x, seq_pos):
        # x shape can be [B, seq_len, num_heads, head_dim]
        # or [B, num_heads, seq_len, head_dim] depending on transpose.
        # Assuming we apply RoPE after transposing to [B, num_heads, seq_len, head_dim]
        pass
        
def apply_rope(xq, xk, freqs_cos, freqs_sin, start_pos: int = 0):
    # xq, xk shape: [Batch, num_heads, seq_len, head_dim]
    seq_len = xq.shape[2]
    
    # slice the freqs to the current sequence length
    cos = freqs_cos[start_pos : start_pos + seq_len]  # [seq_len, dim/2]
    sin = freqs_sin[start_pos : start_pos + seq_len]  # [seq_len, dim/2]
    
    # Expand to match xq and xk shape
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
    
    # Repeat along the last dimension to match head_dim
    cos = torch.repeat_interleave(cos, 2, dim=-1) # [1, 1, seq_len, dim]
    sin = torch.repeat_interleave(sin, 2, dim=-1) # [1, 1, seq_len, dim]

    # Rotary math formulation: 
    # rotate_half swaps adjacent pairs and negates the first
    def rotate_half(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # interleave -x2 and x1
        res = torch.empty_like(x)
        res[..., 0::2] = -x2
        res[..., 1::2] = x1
        return res

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out, xk_out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim must be cleanly divisible by n_heads"
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, start_pos: int = 0, past_key_value: tuple = None, causal: bool = True):
        # x shape: [Batch, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head viewing
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rope(q, k, self.rope.freqs_cos, self.rope.freqs_sin, start_pos)
        
        # KV Cache logic
        if past_key_value is not None:
            # Concatenate past KV with current KV for autoregressive generation
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        current_key_value = (k, v)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply Causal Mask if needed (usually true for training, but specific during cached generation)
        if causal and seq_len > 1:
            # Mask out future tokens
            # Calculate total past sequence length + current length to form the mask correctly
            tot_seq_len = k.shape[2] 
            mask = torch.full((seq_len, tot_seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=tot_seq_len - seq_len + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            
        probs = F.softmax(scores, dim=-1)
        
        # Combine heads
        output = torch.matmul(probs, v) # [B, n_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [B, seq_len, dim]
        
        # Final output projection
        output = self.o_proj(output)
        
        return output, current_key_value


class GatedFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        # SwiGLU generally uses a hidden dim around 4/3 of the calculated 4 * dim, matching LLaMA
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            # Make sure it's a multiple of 256 for better memory alignment if desired, but not strictly needed here
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # Formula: (xW_1 ⊙ Swish(xW_3))W_2
        return self.w2(self.w1(x) * F.silu(self.w3(x)))
