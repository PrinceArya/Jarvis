import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import SmallLanguageModel
from dataset import prepare_dataloader
import time

def train():
    # Hyperparameters
    vocab_size = 4000
    batch_size = 32
    context_length = 128
    dim = 256
    n_layers = 12
    n_heads = 8
    learning_rate = 5e-4
    epochs = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # 1. Load Data
    dataloader, tokenizer = prepare_dataloader(
        split="train", 
        batch_size=batch_size, 
        context_length=context_length, 
        vocab_size=vocab_size
    )
    
    # 2. Init Model
    model = SmallLanguageModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=context_length * 2  # Double context length for max_seq_len to be safe
    ).to(device)
    
    # 3. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    
    # 4. Training Loop
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)
            
            # Compute loss
            # Flatten the logits and targets to [Batch * SeqLen, VocabSize] and [Batch * SeqLen]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Step
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                
            # For demonstration purposes, limit the number of steps to avoid infinite runs locally
            # if step >= 100000:
            #     print("Reached 1000 steps, stopping early for demonstration.")
            #     break
        if(epoch % 10 == 0):
            torch.save(model.state_dict(), f"slm_model_epoch_{epoch}.pt")
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")    

    # Save model
    torch.save(model.state_dict(), "slm_model.pt")
    print("Training finished! Model saved to slm_model.pt")

if __name__ == "__main__":
    train()
