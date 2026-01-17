#!/usr/bin/env python
"""
Pre-training script for PoultryCaduceus.

Usage:
    python scripts/pretrain.py --config configs/pretrain.yaml
    
    # Multi-GPU training
    torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain.yaml
"""

import os
import sys
import argparse
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poultry_caduceus.model import PoultryCaduceus
from poultry_caduceus.config import PoultryCaduceusConfig, TrainingConfig
from poultry_caduceus.tokenizer import DNATokenizer


# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GenomeDataset(Dataset):
    """Dataset for genome pre-training."""
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 65536,
        mask_ratio: float = 0.15
    ):
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.tokenizer = DNATokenizer()
        
        # Load sequences
        logger.info(f"Loading data from {data_path}")
        self.sequences = self._load_sequences(data_path)
        logger.info(f"Loaded {len(self.sequences)} sequences")
    
    def _load_sequences(self, data_path: str):
        """Load sequences from file."""
        sequences = []
        
        if data_path.endswith('.txt'):
            with open(data_path, 'r') as f:
                for line in f:
                    seq = line.strip()
                    if len(seq) >= self.seq_length:
                        sequences.append(seq)
        elif data_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(data_path)
            sequences = df['sequence'].tolist()
        else:
            # Assume FASTA
            from Bio import SeqIO
            for record in SeqIO.parse(data_path, 'fasta'):
                seq = str(record.seq).upper()
                if len(seq) >= self.seq_length:
                    sequences.append(seq)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Random crop if longer than seq_length
        if len(sequence) > self.seq_length:
            start = np.random.randint(0, len(sequence) - self.seq_length)
            sequence = sequence[start:start + self.seq_length]
        
        # Tokenize
        input_ids = self.tokenizer.encode(sequence, return_tensors=None)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create masked input and labels
        input_ids, labels = self._create_mlm_batch(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def _create_mlm_batch(self, input_ids):
        """Create masked language modeling batch."""
        labels = input_ids.clone()
        
        # Create mask
        mask = torch.rand(len(input_ids)) < self.mask_ratio
        
        # 80% [MASK], 10% random, 10% unchanged
        mask_indices = mask.nonzero().squeeze()
        
        if mask_indices.dim() == 0:
            mask_indices = mask_indices.unsqueeze(0)
        
        n_mask = len(mask_indices)
        
        if n_mask > 0:
            # 80% [MASK]
            n_mask_token = int(n_mask * 0.8)
            input_ids[mask_indices[:n_mask_token]] = 5  # [MASK] token
            
            # 10% random
            n_random = int(n_mask * 0.1)
            random_tokens = torch.randint(0, 4, (n_random,))
            input_ids[mask_indices[n_mask_token:n_mask_token + n_random]] = random_tokens
            
            # 10% unchanged (already correct)
        
        # Set labels for non-masked positions to -100 (ignore)
        labels[~mask] = -100
        
        return input_ids, labels


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                output = model(input_ids, labels=labels)
                loss = output.loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_ids, labels=labels)
            loss = output.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(input_ids, labels=labels)
            loss = output.loss
            
            total_loss += loss.item()
            
            # Calculate accuracy on masked tokens
            mask = labels != -100
            predictions = output.logits.argmax(dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Pre-train PoultryCaduceus')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = PoultryCaduceusConfig.from_dict(config_dict['model'])
    train_config = TrainingConfig.from_dict(config_dict['training'])
    
    # Setup device
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create model
    model = PoultryCaduceus(model_config)
    model = model.to(device)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )
    
    logger.info(f"Model parameters: {model.num_parameters:,}")
    
    # Create datasets
    train_dataset = GenomeDataset(
        config_dict['data']['train_path'],
        seq_length=model_config.max_seq_len,
        mask_ratio=config_dict['data'].get('mask_ratio', 0.15)
    )
    
    val_dataset = GenomeDataset(
        config_dict['data']['val_path'],
        seq_length=model_config.max_seq_len,
        mask_ratio=config_dict['data'].get('mask_ratio', 0.15)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_epsilon
    )
    
    # Create scheduler
    total_steps = len(train_loader) * train_config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.learning_rate,
        total_steps=total_steps,
        pct_start=train_config.warmup_steps / total_steps
    )
    
    # Mixed precision
    scaler = GradScaler() if train_config.mixed_precision != "no" else None
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create output directory
    output_dir = Path(config_dict['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, train_config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{train_config.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, train_config
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        logger.info(f"Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'config': config_dict
        }
        
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"New best model saved!")
    
    # Save final model in HuggingFace format
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir / 'final_model')
    
    logger.info(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
