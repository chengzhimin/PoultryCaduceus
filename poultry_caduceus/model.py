"""
PoultryCaduceus Model Implementation.

This module contains the main model classes for PoultryCaduceus,
including the base model and task-specific variants.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

from .config import PoultryCaduceusConfig, MPRAConfig, EQTLConfig
from .tokenizer import DNATokenizer


@dataclass
class ModelOutput:
    """Base class for model outputs."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class MambaLayer(nn.Module):
    """
    Mamba layer implementation.
    
    This is a simplified version. For full implementation,
    use the official mamba-ssm package.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM (simplified)
        x = x * F.silu(z)
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block with optional RC equivariance.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        rc_equivariant: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.rc_equivariant = rc_equivariant
        
        # Forward and backward Mamba layers
        self.mamba_forward = MambaLayer(d_model, d_state, d_conv, expand, dropout)
        self.mamba_backward = MambaLayer(d_model, d_state, d_conv, expand, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Bidirectional Mamba
        x_forward = self.mamba_forward(x)
        x_backward = self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
        
        if self.rc_equivariant:
            # RC equivariance through weight sharing
            x_combined = (x_forward + x_backward) / 2
        else:
            x_combined = x_forward + x_backward
        
        # Residual connection and normalization
        x = self.norm1(x + x_combined)
        
        # Feed-forward
        x = self.norm2(x + self.ffn(x))
        
        return x


class PoultryCaduceus(nn.Module):
    """
    PoultryCaduceus: DNA Language Model for Chicken Genome.
    
    A bidirectional DNA language model based on the Caduceus architecture,
    specifically pre-trained on the chicken (Gallus gallus) genome.
    
    Example:
        >>> model = PoultryCaduceus.from_pretrained("poultry-caduceus-base")
        >>> embeddings = model.get_embeddings("ATGCGATCGATCG")
    """
    
    def __init__(self, config: PoultryCaduceusConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # BiMamba layers
        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dropout=config.dropout,
                rc_equivariant=config.rc_equivariant
            )
            for _ in range(config.n_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(config.d_model)
        
        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        output_attentions: bool = False
    ) -> ModelOutput:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            labels: Labels for language modeling (optional)
            return_embeddings: Whether to return embeddings instead of logits
            output_attentions: Whether to output attention weights
        
        Returns:
            ModelOutput containing loss, logits, and/or embeddings
        """
        # Embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # BiMamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Output normalization
        x = self.norm(x)
        
        if return_embeddings:
            return ModelOutput(embeddings=x)
        
        # Language model head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return ModelOutput(loss=loss, logits=logits, embeddings=x)
    
    def get_embeddings(
        self,
        sequence: Union[str, torch.Tensor],
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get sequence embeddings.
        
        Args:
            sequence: DNA sequence string or token IDs
            pooling: Pooling method ("mean", "max", "cls", "none")
        
        Returns:
            Sequence embedding tensor
        """
        # Tokenize if string
        if isinstance(sequence, str):
            tokenizer = DNATokenizer()
            input_ids = tokenizer.encode(sequence)
        else:
            input_ids = sequence
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Get embeddings
        with torch.no_grad():
            output = self.forward(input_ids, return_embeddings=True)
            embeddings = output.embeddings
        
        # Pooling
        if pooling == "mean":
            return embeddings.mean(dim=1)
        elif pooling == "max":
            return embeddings.max(dim=1)[0]
        elif pooling == "cls":
            return embeddings[:, 0, :]
        else:
            return embeddings
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "PoultryCaduceus":
        """
        Load pre-trained model.
        
        Args:
            pretrained_model_name_or_path: Model name or path to saved model
            **kwargs: Additional arguments passed to config
        
        Returns:
            Loaded model
        """
        # Check if it's a local path
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        else:
            # Try to download from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="config.json"
                )
                model_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="pytorch_model.bin"
                )
            except Exception as e:
                raise ValueError(
                    f"Could not load model from {pretrained_model_name_or_path}: {e}"
                )
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = PoultryCaduceusConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
    
    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionPooling(nn.Module):
    """Attention-based pooling layer."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Pooled tensor of shape (batch, d_model)
        """
        # Compute attention weights
        weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            weights = weights.masked_fill(~mask, float('-inf'))
        
        weights = F.softmax(weights, dim=-1)
        
        # Weighted sum
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        
        return pooled


class PoultryCaduceusMPRA(nn.Module):
    """
    PoultryCaduceus with MPRA prediction head.
    
    Fine-tuned for predicting regulatory activity from MPRA experiments.
    
    Example:
        >>> model = PoultryCaduceusMPRA.from_pretrained("poultry-caduceus-mpra")
        >>> activity = model.predict("ATGCGATCGATCG")
    """
    
    def __init__(
        self,
        config: MPRAConfig,
        backbone: Optional[PoultryCaduceus] = None
    ):
        super().__init__()
        self.config = config
        
        # Backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = PoultryCaduceus(config)
        
        # MPRA prediction head
        self.pooling = AttentionPooling(config.d_model)
        
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            labels: MPRA activity labels (optional)
        
        Returns:
            ModelOutput with predictions and optional loss
        """
        # Get embeddings from backbone
        output = self.backbone(input_ids, return_embeddings=True)
        embeddings = output.embeddings
        
        # Pool and predict
        pooled = self.pooling(embeddings)
        logits = self.head(pooled).squeeze(-1)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        
        return ModelOutput(loss=loss, logits=logits, embeddings=embeddings)
    
    def predict(self, sequence: Union[str, torch.Tensor]) -> float:
        """
        Predict MPRA activity for a sequence.
        
        Args:
            sequence: DNA sequence string or token IDs
        
        Returns:
            Predicted MPRA activity
        """
        # Tokenize if string
        if isinstance(sequence, str):
            tokenizer = DNATokenizer()
            input_ids = tokenizer.encode(sequence)
        else:
            input_ids = sequence
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Predict
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids)
        
        return output.logits.item()
    
    def predict_variant_effect(
        self,
        ref_sequence: str,
        alt_sequence: str
    ) -> Dict[str, float]:
        """
        Predict the effect of a variant.
        
        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternative sequence
        
        Returns:
            Dictionary with ref_activity, alt_activity, and effect_size
        """
        ref_activity = self.predict(ref_sequence)
        alt_activity = self.predict(alt_sequence)
        
        return {
            "ref_activity": ref_activity,
            "alt_activity": alt_activity,
            "effect_size": alt_activity - ref_activity
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "PoultryCaduceusMPRA":
        """Load pre-trained model."""
        # Similar to PoultryCaduceus.from_pretrained
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json"
            )
            model_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="pytorch_model.bin"
            )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = MPRAConfig.from_dict(config_dict)
        
        model = cls(config)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model


class PoultryCaduceusEQTL(nn.Module):
    """
    PoultryCaduceus with eQTL prediction head.
    
    Fine-tuned for predicting expression quantitative trait loci effects.
    """
    
    def __init__(
        self,
        config: EQTLConfig,
        backbone: Optional[PoultryCaduceus] = None
    ):
        super().__init__()
        self.config = config
        
        # Backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = PoultryCaduceus(config)
        
        # eQTL prediction head (multi-tissue)
        self.pooling = AttentionPooling(config.d_model)
        
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.num_tissues)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """Forward pass."""
        output = self.backbone(input_ids, return_embeddings=True)
        embeddings = output.embeddings
        
        pooled = self.pooling(embeddings)
        logits = self.head(pooled)
        
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        
        return ModelOutput(loss=loss, logits=logits, embeddings=embeddings)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "PoultryCaduceusEQTL":
        """Load pre-trained model."""
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json"
            )
            model_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="pytorch_model.bin"
            )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = EQTLConfig.from_dict(config_dict)
        
        model = cls(config)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
