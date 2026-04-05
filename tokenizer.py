import json
import os
from collections import Counter
from typing import List, Dict, Tuple

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate the frequency of consecutive pairs in a sequence of ids."""
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace all consecutive occurrences of `pair` with the new token `idx`."""
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BPETokenizer:
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        # Initial vocabulary of raw 256 bytes
        self.vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    
    def train(self, text: str, vocab_size: int, show_progress: bool = False):
        """Train the tokenizer using the BPE algorithm."""
        num_merges = vocab_size - 256
        if num_merges <= 0:
            return

        # Start with raw bytes of the utf-8 encoded text
        ids = list(text.encode("utf-8"))
        
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats: 
                break
            
            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            if show_progress:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} (count {stats[pair]})")
            
            # Update IDs by performing the merge
            ids = merge(ids, pair, idx)
            
            # Save the merge rule and update vocabulary
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token ids."""
        ids = list(text.encode("utf-8"))
        # We need to iteratively apply the merges in the order they were learned!
        # Wait, the best way to encode is actually to continually find the pair that exists in merges with the lowest id
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair that has the minimum merge index (meaning it was merged earliest)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # No more applicable merges
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into a string."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # We use replace to handle any invalid utf-8 sequences
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_prefix: str):
        """Save the model's merges and vocab."""
        # Convert merges to a string-keyed dict to save as JSON
        merges_to_save = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(f"{file_prefix}_merges.json", "w") as f:
            json.dump(merges_to_save, f)
            
    def load(self, file_prefix: str):
        """Load the model's merges and recreate the vocab."""
        with open(f"{file_prefix}_merges.json", "r") as f:
            merges_loaded = json.load(f)
            
        self.merges = {}
        for k_str, v in merges_loaded.items():
            p0, p1 = map(int, k_str.split(","))
            self.merges[(p0, p1)] = v
            
        # Reconstruct vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # The merges dict is unordered when loaded from JSON, but we can infer the order because
        # idx goes from 256 onwards sequentially.
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda item: item[1]):
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
