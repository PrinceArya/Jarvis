import torch
import torch.nn as nn
from modules import MultiHeadAttention, GatedFFN

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0, past_key_value: tuple = None, causal: bool = True):
        # Pre-LN: Apply LayerNorm before attention
        norm_x = self.ln_1(x)
        
        # MHA block
        attn_out, current_kv = self.attn(norm_x, start_pos, past_key_value, causal=causal)
        
        # Residual connection
        x = x + attn_out
        
        # Pre-LN: Apply LayerNorm before FFN
        norm_x = self.ln_2(x)
        
        # FFN block
        ffn_out = self.ffn(norm_x)
        
        # Residual connection
        x = x + ffn_out
        
        return x, current_kv

class SmallLanguageModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        dim: int = 256, 
        n_layers: int = 4, 
        n_heads: int = 8, 
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # Token Embedding
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, max_seq_len) for _ in range(n_layers)
        ])
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # Language Modeling Head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, past_key_values: list = None, causal: bool = True):
        """
        Forward pass for the SLM.
        
        Args:
            input_ids: [Batch, SeqLen]
            start_pos: the sequence start position (for RoPE offset) when doing KV caching
            past_key_values: list of (key, value) cache pairs for each layer
            causal: whether to apply a causal attention mask
            
        Returns:
            logits: [Batch, SeqLen, VocabSize]
            next_past_key_values: list of updated (key, value) cache pairs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embeddings(input_ids)
        
        next_past_key_values = []
        
        # Pass through all transformer blocks
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            x, current_kv = layer(x, start_pos, layer_past_kv, causal=causal)
            next_past_key_values.append(current_kv)
            
        # Final layer norm
        x = self.norm(x)
        
        # LM Head to get logits
        logits = self.lm_head(x)
        
        return logits, next_past_key_values
