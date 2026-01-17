"""
DNA Tokenizer for PoultryCaduceus.
"""

import torch
import numpy as np
from typing import Union, List, Optional


class DNATokenizer:
    """
    Tokenizer for DNA sequences.
    
    Converts DNA sequences to integer tokens and vice versa.
    
    Vocabulary:
        - A: 0
        - C: 1
        - G: 2
        - T: 3
        - N: 4 (unknown/ambiguous)
        - [MASK]: 5 (for masked language modeling)
        - [PAD]: 6 (for padding, optional)
    
    Example:
        >>> tokenizer = DNATokenizer()
        >>> tokens = tokenizer.encode("ATGCN")
        >>> print(tokens)
        tensor([0, 3, 2, 1, 4])
        >>> sequence = tokenizer.decode(tokens)
        >>> print(sequence)
        'ATGCN'
    """
    
    # Vocabulary mapping
    NUCLEOTIDE_TO_ID = {
        'A': 0, 'a': 0,
        'C': 1, 'c': 1,
        'G': 2, 'g': 2,
        'T': 3, 't': 3,
        'N': 4, 'n': 4,
        'U': 3, 'u': 3,  # RNA support (U -> T)
    }
    
    ID_TO_NUCLEOTIDE = {
        0: 'A',
        1: 'C',
        2: 'G',
        3: 'T',
        4: 'N',
        5: '[MASK]',
        6: '[PAD]',
    }
    
    # Special tokens
    MASK_TOKEN_ID = 5
    PAD_TOKEN_ID = 6
    UNK_TOKEN_ID = 4  # Unknown maps to N
    
    # Complement mapping for reverse complement
    COMPLEMENT = {
        'A': 'T', 'T': 'A',
        'C': 'G', 'G': 'C',
        'N': 'N',
        'a': 't', 't': 'a',
        'c': 'g', 'g': 'c',
        'n': 'n',
    }
    
    COMPLEMENT_ID = {
        0: 3,  # A -> T
        1: 2,  # C -> G
        2: 1,  # G -> C
        3: 0,  # T -> A
        4: 4,  # N -> N
    }
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        """
        Initialize the tokenizer.
        
        Args:
            max_length: Maximum sequence length (None for no limit)
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences longer than max_length
            return_tensors: Return type ("pt" for PyTorch, "np" for NumPy)
        """
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        
        self.vocab_size = 7  # A, C, G, T, N, [MASK], [PAD]
    
    def encode(
        self,
        sequence: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False
    ) -> Union[torch.Tensor, np.ndarray, List[int]]:
        """
        Encode DNA sequence(s) to token IDs.
        
        Args:
            sequence: DNA sequence string or list of sequences
            max_length: Override default max_length
            padding: Override default padding
            truncation: Override default truncation
            return_tensors: Override default return type
            add_special_tokens: Whether to add special tokens (not used currently)
        
        Returns:
            Token IDs as tensor, array, or list
        """
        max_length = max_length or self.max_length
        padding = padding if padding is not None else self.padding
        truncation = truncation if truncation is not None else self.truncation
        return_tensors = return_tensors or self.return_tensors
        
        # Handle batch input
        if isinstance(sequence, list):
            return self._encode_batch(
                sequence, max_length, padding, truncation, return_tensors
            )
        
        # Single sequence
        tokens = self._encode_single(sequence)
        
        # Truncation
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Padding
        if padding and max_length and len(tokens) < max_length:
            tokens = tokens + [self.PAD_TOKEN_ID] * (max_length - len(tokens))
        
        # Convert to output format
        if return_tensors == "pt":
            return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        elif return_tensors == "np":
            return np.array(tokens, dtype=np.int64)
        else:
            return tokens
    
    def _encode_single(self, sequence: str) -> List[int]:
        """Encode a single sequence to token IDs."""
        return [self.NUCLEOTIDE_TO_ID.get(nt, self.UNK_TOKEN_ID) for nt in sequence]
    
    def _encode_batch(
        self,
        sequences: List[str],
        max_length: Optional[int],
        padding: bool,
        truncation: bool,
        return_tensors: str
    ) -> Union[torch.Tensor, np.ndarray, List[List[int]]]:
        """Encode a batch of sequences."""
        encoded = [self._encode_single(seq) for seq in sequences]
        
        # Truncation
        if truncation and max_length:
            encoded = [tokens[:max_length] for tokens in encoded]
        
        # Padding
        if padding:
            if max_length:
                target_length = max_length
            else:
                target_length = max(len(tokens) for tokens in encoded)
            
            encoded = [
                tokens + [self.PAD_TOKEN_ID] * (target_length - len(tokens))
                for tokens in encoded
            ]
        
        # Convert to output format
        if return_tensors == "pt":
            return torch.tensor(encoded, dtype=torch.long)
        elif return_tensors == "np":
            return np.array(encoded, dtype=np.int64)
        else:
            return encoded
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, np.ndarray, List[int]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to DNA sequence(s).
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip [MASK] and [PAD] tokens
        
        Returns:
            DNA sequence string or list of sequences
        """
        # Convert to list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Handle batch
        if isinstance(token_ids[0], list):
            return [self._decode_single(ids, skip_special_tokens) for ids in token_ids]
        
        return self._decode_single(token_ids, skip_special_tokens)
    
    def _decode_single(self, token_ids: List[int], skip_special_tokens: bool) -> str:
        """Decode a single sequence."""
        nucleotides = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.MASK_TOKEN_ID, self.PAD_TOKEN_ID]:
                continue
            nucleotides.append(self.ID_TO_NUCLEOTIDE.get(token_id, 'N'))
        return ''.join(nucleotides)
    
    def reverse_complement(
        self,
        sequence: Union[str, torch.Tensor, np.ndarray]
    ) -> Union[str, torch.Tensor, np.ndarray]:
        """
        Get the reverse complement of a DNA sequence.
        
        Args:
            sequence: DNA sequence (string or token IDs)
        
        Returns:
            Reverse complement in the same format as input
        """
        if isinstance(sequence, str):
            return ''.join(self.COMPLEMENT.get(nt, 'N') for nt in sequence[::-1])
        
        # Token IDs
        if isinstance(sequence, torch.Tensor):
            rc_ids = torch.tensor([self.COMPLEMENT_ID.get(int(t), 4) for t in sequence.flip(-1)])
            return rc_ids
        elif isinstance(sequence, np.ndarray):
            rc_ids = np.array([self.COMPLEMENT_ID.get(int(t), 4) for t in sequence[::-1]])
            return rc_ids
        else:
            return [self.COMPLEMENT_ID.get(t, 4) for t in sequence[::-1]]
    
    def __call__(
        self,
        sequence: Union[str, List[str]],
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray, List[int]]:
        """Shortcut for encode()."""
        return self.encode(sequence, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"DNATokenizer(vocab_size={self.vocab_size}, "
            f"max_length={self.max_length}, "
            f"padding={self.padding}, "
            f"truncation={self.truncation})"
        )
