import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizer import BPETokenizer

class WikiTextDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        # We can extract (len(token_ids) - context_length) number of sequences
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx):
        # The input is context_length tokens
        x = self.token_ids[idx : idx + self.context_length]
        # The label is the same context_length shifted by 1
        y = self.token_ids[idx + 1 : idx + self.context_length + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def prepare_dataloader(split: str = "train", batch_size: int = 16, context_length: int = 256, vocab_size: int = 1000):
    """
    Downloads WikiText-2, trains or loads the BPE tokenizer, tokenizes the requested split,
    and returns a PyTorch DataLoader.
    """
    print(f"Loading empathetic_dialogues dataset ({split} split)...")
    dataset = load_dataset("empathetic_dialogues", split=split, trust_remote_code=True)
    
    text_data = "\n".join([str(utt).replace("_comma_", ",") for utt in dataset["utterance"]])
    
    # Try loading existing tokenizer to avoid retraining
    tokenizer = BPETokenizer()
    tokenizer_path = "empathetic_dialogues_tokenizer"
    
    if os.path.exists(f"{tokenizer_path}_merges.json"):
        print(f"Loading cached tokenizer from {tokenizer_path}_merges.json...")
        tokenizer.load(tokenizer_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={vocab_size}) on training data...")
        # Always train on the train split to prevent data leakage from validation/test
        train_dataset = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
        train_text = "\n".join([str(utt).replace("_comma_", ",") for utt in train_dataset["utterance"]])
        vocab_train_text = train_text[:500000]
        tokenizer.train(vocab_train_text, vocab_size, show_progress=False)
        tokenizer.save(tokenizer_path)
        print("Tokenizer trained and saved.")

    print("Encoding dataset to token IDs... This may take a moment.")
    if split == "validation":
        if len(text_data) > 20000:
            print("Truncating validation text data to 20K characters for faster BPE encoding...")
            text_data = text_data[:20000]
    else:
        if len(text_data) > 500000:
            print(f"Truncating {split} text data to 500K characters for faster BPE encoding...")
            text_data = text_data[:500000]
        
    token_ids = tokenizer.encode(text_data)
    print(f"Encoded {len(token_ids)} tokens.")
    
    # Create the dataset and dataloader
    pt_dataset = WikiTextDataset(token_ids, context_length)
    dataloader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader, tokenizer
