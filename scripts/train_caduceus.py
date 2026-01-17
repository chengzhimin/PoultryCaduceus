#!/usr/bin/env python
"""
Training script for PoultryCaduceus using the original Caduceus codebase.

This script wraps the Caduceus training pipeline for chicken genome pre-training.

Usage:
    # Clone Caduceus first
    git clone https://github.com/kuleshov-group/caduceus.git
    cd caduceus
    pip install -e .
    
    # Then run training
    python scripts/train_caduceus.py \
        --data_dir /path/to/chicken_pretrain_data \
        --output_dir /path/to/checkpoints \
        --model_name caduceus-ps_seqlen-131k_d_model-256_n_layer-16

For H100/H200 servers without internet:
    1. Prepare data on Colab using data_preparation.ipynb
    2. Download the tar.gz file
    3. Upload to server and extract
    4. Run this script
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path


def check_caduceus_installation():
    """Check if Caduceus is installed."""
    try:
        import caduceus
        print(f"✓ Caduceus installed: {caduceus.__file__}")
        return True
    except ImportError:
        print("✗ Caduceus not found")
        print("\nPlease install Caduceus:")
        print("  git clone https://github.com/kuleshov-group/caduceus.git")
        print("  cd caduceus")
        print("  pip install -e .")
        return False


def check_mamba_installation():
    """Check if Mamba SSM is installed."""
    try:
        import mamba_ssm
        print(f"✓ Mamba SSM installed: {mamba_ssm.__file__}")
        return True
    except ImportError:
        print("✗ Mamba SSM not found")
        print("\nPlease install Mamba SSM:")
        print("  pip install mamba-ssm")
        print("  # or build from source for better performance")
        return False


def create_caduceus_config(args) -> dict:
    """Create configuration for Caduceus training."""
    
    config = {
        # Model
        "model": {
            "name": args.model_name,
            "d_model": args.d_model,
            "n_layer": args.n_layers,
            "vocab_size": 12,  # Caduceus vocabulary
            "complement_map": {
                "A": "T", "T": "A",
                "C": "G", "G": "C",
                "N": "N"
            },
            "rc_equivariant": True,
        },
        
        # Data
        "data": {
            "train_path": str(args.data_dir / "train.txt"),
            "val_path": str(args.data_dir / "val.txt"),
            "seq_length": args.seq_length,
            "rc_augmentation": True,
        },
        
        # Training
        "training": {
            "output_dir": str(args.output_dir),
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "num_epochs": args.epochs,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "mixed_precision": args.mixed_precision,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "logging_steps": args.logging_steps,
        },
        
        # Hardware
        "hardware": {
            "num_workers": args.num_workers,
            "pin_memory": True,
        }
    }
    
    return config


def run_training_native(config: dict, args):
    """Run training using native PyTorch (without Caduceus CLI)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm
    import h5py
    import numpy as np
    
    print("\n" + "="*60)
    print("Starting PoultryCaduceus Training")
    print("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA not available, using CPU (will be slow)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("\nLoading model...")
    try:
        from caduceus.modeling_caduceus import CaduceusForMaskedLM, CaduceusConfig
        
        model_config = CaduceusConfig(
            d_model=config["model"]["d_model"],
            n_layer=config["model"]["n_layer"],
            vocab_size=config["model"]["vocab_size"],
            complement_map=config["model"]["complement_map"],
            rcps=config["model"]["rc_equivariant"],
        )
        model = CaduceusForMaskedLM(model_config)
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except ImportError:
        print("Caduceus not available, using simplified model...")
        # Fallback to simplified model
        from poultry_caduceus.model import PoultryCaduceus
        from poultry_caduceus.config import PoultryCaduceusConfig
        
        model_config = PoultryCaduceusConfig(
            d_model=config["model"]["d_model"],
            n_layers=config["model"]["n_layer"],
        )
        model = PoultryCaduceus(model_config)
        print(f"✓ Simplified model created: {model.num_parameters:,} parameters")
    
    model = model.to(device)
    
    # Create dataset
    class GenomeDataset(Dataset):
        def __init__(self, data_path, seq_length):
            self.seq_length = seq_length
            self.char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
            
            if data_path.endswith('.h5'):
                with h5py.File(data_path, 'r') as f:
                    self.sequences = f['sequences'][:]
                    self.n_seqs = len(self.sequences)
                self.format = 'h5'
            else:
                with open(data_path, 'r') as f:
                    self.sequences = [line.strip() for line in f if line.strip()]
                self.n_seqs = len(self.sequences)
                self.format = 'text'
            
            print(f"  Loaded {self.n_seqs:,} sequences from {data_path}")
        
        def __len__(self):
            return self.n_seqs
        
        def __getitem__(self, idx):
            if self.format == 'h5':
                seq = self.sequences[idx]
            else:
                seq = self.sequences[idx]
                seq = np.array([self.char_to_int.get(c, 4) for c in seq], dtype=np.uint8)
            
            # Create masked input
            input_ids = torch.tensor(seq, dtype=torch.long)
            labels = input_ids.clone()
            
            # Random masking
            mask = torch.rand(len(input_ids)) < 0.15
            input_ids[mask] = 5  # Mask token
            labels[~mask] = -100  # Ignore non-masked
            
            return {"input_ids": input_ids, "labels": labels}
    
    print("\nLoading data...")
    train_dataset = GenomeDataset(config["data"]["train_path"], config["data"]["seq_length"])
    val_dataset = GenomeDataset(config["data"]["val_path"], config["data"]["seq_length"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"]
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Scheduler
    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    # Mixed precision
    scaler = GradScaler() if config["training"]["mixed_precision"] != "no" else None
    
    # Training loop
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Total steps: {total_steps:,}")
    
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                optimizer.step()
            
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            # Save checkpoint
            if global_step % config["training"]["save_steps"] == 0:
                checkpoint_path = output_dir / f"checkpoint-{global_step}"
                checkpoint_path.mkdir(exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": config,
                }, checkpoint_path / "pytorch_model.bin")
                print(f"\n  Saved checkpoint to {checkpoint_path}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        epoch_loss /= len(train_loader)
        
        print(f"\nEpoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model"
            best_path.mkdir(exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
            }, best_path / "pytorch_model.bin")
            print(f"  New best model saved! val_loss={val_loss:.4f}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train PoultryCaduceus')
    
    # Data
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Directory containing train.txt/val.txt or train.h5/val.h5')
    parser.add_argument('--output_dir', type=Path, default=Path('checkpoints/pretrain'),
                        help='Output directory for checkpoints')
    
    # Model
    parser.add_argument('--model_name', type=str, 
                        default='caduceus-ps_seqlen-131k_d_model-256_n_layer-16',
                        help='Caduceus model name')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model hidden dimension')
    parser.add_argument('--n_layers', type=int, default=16,
                        help='Number of layers')
    parser.add_argument('--seq_length', type=int, default=65536,
                        help='Sequence length')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Max training steps (-1 for unlimited)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training')
    
    # Checkpointing
    parser.add_argument('--save_steps', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Check installations
    print("Checking dependencies...")
    caduceus_ok = check_caduceus_installation()
    mamba_ok = check_mamba_installation()
    
    if not mamba_ok:
        print("\nWarning: Mamba SSM not installed. Training may be slow.")
    
    # Create config
    config = create_caduceus_config(args)
    
    # Save config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    run_training_native(config, args)


if __name__ == '__main__':
    main()
