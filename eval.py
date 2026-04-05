import os
import torch
import torch.nn.functional as F
from model import SmallLanguageModel
from dataset import prepare_dataloader
from tokenizer import BPETokenizer

def evaluate_perplexity(model, dataloader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            
            # CrossEntropyLoss expects standard classification shapes 
            # (Batch*SeqLen, VocabSize) and (Batch*SeqLen)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
            
            # Limit eval for demo speed
            if total_steps >= 200:
                break
                
    avg_loss = total_loss / total_steps
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    return perplexity.item()

def generate_text(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        # Default starting token if prompt is empty or fails encoding
        tokens = [0]
        
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device) # [1, SeqLen]
    
    print(f"\nGenerating text with KV cache for prompt: '{prompt}'")
    
    # 1. Forward pass full prompt to fill KV cache
    with torch.no_grad():
        logits, past_key_values = model(input_ids)
        
    # The next token prediction comes from the last position of logits
    next_token_logits = logits[0, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).item()
    tokens.append(next_token)
    
    # The starting position for the next single-token pass
    start_pos = input_ids.shape[1]
    
    # 2. Autoregressive generation using KV cache
    for _ in range(max_new_tokens - 1): # We already generated 1
        # Prepare the single new token
        input_id = torch.tensor([[next_token]], dtype=torch.long, device=device) # [1, 1]
        
        with torch.no_grad():
            # Pass ONLY the single token, the start_pos, and the cached KVs
            # Important: set causal=False because we're only passing sequence length of 1
            logits, past_key_values = model(
                input_id, 
                start_pos=start_pos, 
                past_key_values=past_key_values,
                causal=False 
            )
            
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        tokens.append(next_token)
        
        start_pos += 1

    generated_text = tokenizer.decode(tokens)
    print(f"Result:\n{generated_text}")
    return generated_text

def main():
    # Setup
    vocab_size = 4000
    context_length = 128
    dim = 256
    n_layers = 12
    n_heads = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tokenizer directly since it should be cached by now
    tokenizer = BPETokenizer()
    if os.path.exists("empathetic_dialogues_tokenizer_merges.json"):
        tokenizer.load("empathetic_dialogues_tokenizer")
    else:
        print("Tokenizer not found. Run train.py first to prepare data and tokenizer.")
        return

    # Check for model weights
    if not os.path.exists("slm_model_epoch_10.pt"):
        print("Model file 'slm_model.pt' not found. Please run train.py to train the model first.")
        return
        
    # Init Model
    model = SmallLanguageModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=context_length * 2
    ).to(device)
    
    model.load_state_dict(torch.load("slm_model_epoch_10.pt", map_location=device))
    
    # Prepare DataLoader for Validation
    val_dataloader, _ = prepare_dataloader(
        split="validation", # DailyDialog validation set
        batch_size=8, 
        context_length=context_length, 
        vocab_size=vocab_size
    )

    evaluate_perplexity(model, val_dataloader, device, vocab_size)
    
    generate_text(model, tokenizer, prompt="The history of ", max_new_tokens=70, device=device)

if __name__ == "__main__":
    main()
