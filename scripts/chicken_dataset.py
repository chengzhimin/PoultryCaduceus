#!/usr/bin/env python3
"""
Chicken Caduceus Dataset Class (v5 Fixed Version)

Fixes: 
1. HDF5 key changed from 'tokens' to 'sequences' (matching your data format)
2. Support auto-detection of key names
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from typing import Optional, Tuple

# DNA Vocabulary (consistent with official Caduceus, vocab_size=16)
DNA_VOCAB = {
    '[PAD]': 0,
    '[UNK]': 1,
    '[CLS]': 2,
    '[SEP]': 3,
    '[MASK]': 4,
    'N': 5,
    '.': 6,
    'A': 7,
    'C': 8,
    'G': 9,
    'T': 10,
    'R': 11,
    'Y': 12,
    'S': 13,
    'W': 14,
    'K': 15,
}

ID_TO_TOKEN = {v: k for k, v in DNA_VOCAB.items()}

# Complement base mapping
COMPLEMENT_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 10,  # A -> T
    8: 9,   # C -> G
    9: 8,   # G -> C
    10: 7,  # T -> A
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
}


def reverse_complement(token_ids: np.ndarray) -> np.ndarray:
    """Compute reverse complement sequence"""
    rc = token_ids[::-1].copy()
    for i in range(len(rc)):
        rc[i] = COMPLEMENT_MAP.get(int(rc[i]), rc[i])
    return rc


class ChickenGenomeDataset(Dataset):
    """
    Chicken Genome Dataset (HDF5 format) - v5 Fixed Version
    
    Supported HDF5 formats:
    - 'sequences': (N, seq_len) uint8, encoding A=7, C=8, G=9, T=10, N=11
    - or 'tokens': same as above
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 131072,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        rc_aug: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.rc_aug = rc_aug
        
        self.h5_file = h5py.File(self.data_path, 'r')
        
        # Auto-detect data key (supports 'sequences' or 'tokens')
        if 'sequences' in self.h5_file:
            self.data_key = 'sequences'
        elif 'tokens' in self.h5_file:
            self.data_key = 'tokens'
        else:
            available_keys = list(self.h5_file.keys())
            raise KeyError(f"Cannot find 'sequences' or 'tokens' in HDF5 file. Available keys: {available_keys}")
        
        self.tokens = self.h5_file[self.data_key]
        
        self.n_samples = len(self.tokens)
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)
        
        # Detect data encoding range
        sample_data = self.tokens[0][:100]
        unique_vals = np.unique(sample_data)
        
        print(f"Loading dataset: {self.data_path}")
        print(f"  Data key: '{self.data_key}'")
        print(f"  Number of samples: {self.n_samples}")
        print(f"  Original sequence length: {self.tokens.shape[1]}")
        print(f"  Target sequence length: {self.seq_length}")
        print(f"  Data encoding range: {unique_vals.min()}-{unique_vals.max()}")
        print(f"  MLM: {self.mlm} (p={self.mlm_probability})")
        print(f"  RC augmentation: {self.rc_aug}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get token sequence
        tokens = np.array(self.tokens[idx], dtype=np.int64)
        
        # Truncate or pad to specified length
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        elif len(tokens) < self.seq_length:
            padding = np.zeros(self.seq_length - len(tokens), dtype=np.int64)
            tokens = np.concatenate([tokens, padding])
        
        # Ensure all tokens are in valid range [0, 15]
        tokens = np.clip(tokens, 0, 15)
        
        # RC data augmentation (50% probability)
        if self.rc_aug and random.random() < 0.5:
            tokens = reverse_complement(tokens)
        
        # MLM
        if self.mlm:
            input_ids, labels = self._apply_mlm(tokens)
        else:
            input_ids = tokens.copy()
            labels = tokens.copy()
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    
    def _apply_mlm(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Masked Language Modeling
        
        Caduceus model uses pad_token_id (0) as ignore_index
        So we need:
        - Masked positions: labels = original token (for loss computation)
        - Unmasked positions: labels = 0 (pad_token_id, ignored)
        """
        input_ids = tokens.copy()
        # Initialize labels to 0 (pad_token_id), these positions will be ignored
        labels = np.zeros_like(tokens)
        
        # Only mask valid bases (A=7, C=8, G=9, T=10)
        # Note: N=5 or N=11 do not participate in masking
        valid_mask = (tokens >= 7) & (tokens <= 10)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return input_ids, labels
        
        # Randomly select 15% of positions for masking
        n_mask = max(1, int(len(valid_indices) * self.mlm_probability))
        mask_indices = np.random.choice(valid_indices, size=n_mask, replace=False)
        
        for idx in mask_indices:
            # Save original token as label (for loss computation)
            labels[idx] = tokens[idx]
            
            rand = random.random()
            if rand < 0.8:
                # 80%: Replace with [MASK] (token_id=4)
                input_ids[idx] = 4
            elif rand < 0.9:
                # 10%: Replace with random base (A=7, C=8, G=9, T=10)
                input_ids[idx] = random.choice([7, 8, 9, 10])
            # 10%: Keep unchanged
        
        return input_ids, labels
    
    def close(self):
        self.h5_file.close()


# Test code
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chicken_dataset.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    print("Testing ChickenGenomeDataset v5...")
    
    dataset = ChickenGenomeDataset(
        data_path=data_path,
        seq_length=65536,
        mlm=True,
        mlm_probability=0.15,
        rc_aug=True,
        max_samples=10,
    )
    
    sample = dataset[0]
    input_ids = sample['input_ids']
    labels = sample['labels']
    
    print(f"\ninput_ids shape: {input_ids.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"input_ids range: [{input_ids.min()}, {input_ids.max()}]")
    print(f"labels range: [{labels.min()}, {labels.max()}]")
    print(f"Number of non-zero labels (masked): {(labels > 0).sum()}")
    print(f"\ninput_ids[:50]: {input_ids[:50].tolist()}")
    print(f"labels[:50]: {labels[:50].tolist()}")
    
    # Verify labels are in valid range
    assert labels.min() >= 0, f"labels min {labels.min()} < 0"
    assert labels.max() <= 15, f"labels max {labels.max()} > 15"
    
    print("\n✓ Test passed!")
