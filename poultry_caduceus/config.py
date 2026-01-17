"""
Configuration classes for PoultryCaduceus models.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import yaml


@dataclass
class PoultryCaduceusConfig:
    """
    Configuration class for PoultryCaduceus model.
    
    Args:
        vocab_size: Size of vocabulary (default: 6 for A, C, G, T, N, [MASK])
        d_model: Hidden dimension size
        n_layers: Number of BiMamba layers
        d_state: Mamba state dimension
        d_conv: Convolution kernel size in Mamba
        expand: Expansion factor for FFN
        max_seq_len: Maximum sequence length
        rc_equivariant: Whether to use reverse complement equivariance
        dropout: Dropout probability
        initializer_range: Standard deviation for weight initialization
    """
    
    # Model architecture
    vocab_size: int = 6
    d_model: int = 256
    n_layers: int = 8
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    max_seq_len: int = 65536
    rc_equivariant: bool = True
    
    # Regularization
    dropout: float = 0.1
    
    # Initialization
    initializer_range: float = 0.02
    
    # Model type
    model_type: str = "poultry_caduceus"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "max_seq_len": self.max_seq_len,
            "rc_equivariant": self.rc_equivariant,
            "dropout": self.dropout,
            "initializer_range": self.initializer_range,
            "model_type": self.model_type,
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PoultryCaduceusConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k != "model_type"})
    
    @classmethod
    def load(cls, path: str) -> "PoultryCaduceusConfig":
        """Load configuration from file."""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        return f"PoultryCaduceusConfig({self.to_dict()})"


@dataclass
class MPRAConfig(PoultryCaduceusConfig):
    """Configuration for MPRA fine-tuned model."""
    
    # MPRA head configuration
    head_hidden_dim: int = 256
    head_dropout: float = 0.1
    head_num_layers: int = 2
    
    # Task configuration
    task: str = "mpra_regression"
    
    model_type: str = "poultry_caduceus_mpra"


@dataclass
class EQTLConfig(PoultryCaduceusConfig):
    """Configuration for eQTL fine-tuned model."""
    
    # eQTL head configuration
    head_hidden_dim: int = 256
    head_dropout: float = 0.1
    num_tissues: int = 6
    
    # Task configuration
    task: str = "eqtl_prediction"
    
    model_type: str = "poultry_caduceus_eqtl"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    scheduler: str = "cosine"
    warmup_steps: int = 10000
    
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
