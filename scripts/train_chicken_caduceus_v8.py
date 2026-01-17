#!/usr/bin/env python3
"""
Chicken Caduceus Training Script (v8)
- Built-in fixed ChickenGenomeDataset (supports 'sequences' and 'tokens' keys)
- BF16 mixed precision training
- Distributed training support
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)

import yaml

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed, logging disabled")


# =============================================================================
# DNA Vocabulary (consistent with official Caduceus, vocab_size=16)
# =============================================================================
DNA_VOCAB = {
    '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
    'N': 5, '.': 6, 'A': 7, 'C': 8, 'G': 9, 'T': 10,
    'R': 11, 'Y': 12, 'S': 13, 'W': 14, 'K': 15,
}

# Complement base mapping
COMPLEMENT_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 10, 8: 9, 9: 8, 10: 7,  # A<->T, C<->G
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
}


def reverse_complement(token_ids: np.ndarray) -> np.ndarray:
    """Compute reverse complement sequence"""
    rc = token_ids[::-1].copy()
    for i in range(len(rc)):
        rc[i] = COMPLEMENT_MAP.get(int(rc[i]), rc[i])
    return rc


# =============================================================================
# Dataset (v5 fixed version - built-in)
# =============================================================================
class ChickenGenomeDataset(Dataset):
    """
    Chicken Genome Dataset (HDF5 format)
    
    Supported HDF5 formats:
    - 'sequences': (N, seq_len) uint8, encoding A=7, C=8, G=9, T=10, N=11
    - or 'tokens': same as above
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 65536,
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
        
        # Key fix: auto-detect data key
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
        print(f"  Encoding range: {unique_vals.min()}-{unique_vals.max()}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        tokens = np.array(self.tokens[idx], dtype=np.int64)
        
        # Truncate or pad
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        elif len(tokens) < self.seq_length:
            padding = np.zeros(self.seq_length - len(tokens), dtype=np.int64)
            tokens = np.concatenate([tokens, padding])
        
        tokens = np.clip(tokens, 0, 15)
        
        # RC augmentation
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
        input_ids = tokens.copy()
        labels = np.zeros_like(tokens)  # 0 = ignore
        
        # Only mask valid bases (A=7, C=8, G=9, T=10)
        valid_mask = (tokens >= 7) & (tokens <= 10)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return input_ids, labels
        
        n_mask = max(1, int(len(valid_indices) * self.mlm_probability))
        mask_indices = np.random.choice(valid_indices, size=n_mask, replace=False)
        
        for idx in mask_indices:
            labels[idx] = tokens[idx]
            rand = random.random()
            if rand < 0.8:
                input_ids[idx] = 4  # [MASK]
            elif rand < 0.9:
                input_ids[idx] = random.choice([7, 8, 9, 10])
        
        return input_ids, labels
    
    def close(self):
        self.h5_file.close()


# =============================================================================
# Trainer
# =============================================================================
def setup_distributed():
    """Setup distributed training environment"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is main process"""
    return rank == 0


def print_rank0(msg, rank=0):
    """Print only on main process"""
    if is_main_process(rank):
        print(msg)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, scheduler, step, loss, config, output_dir, rank):
    """Save model checkpoint"""
    if not is_main_process(rank):
        return
    
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(checkpoint_dir)
    
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config,
    }, checkpoint_dir / 'training_state.pt')
    
    print(f"Checkpoint saved: {checkpoint_dir}")


class Trainer:
    """Main trainer class for Chicken Caduceus"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        print_rank0("\n" + "="*60, self.rank)
        print_rank0("Chicken Caduceus Trainer v8", self.rank)
        print_rank0("="*60, self.rank)
        print_rank0(f"Rank: {self.rank}/{self.world_size} | Device: {self.device}", self.rank)
        
        self.output_dir = Path(config['training']['output_dir'])
        if is_main_process(self.rank):
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_amp = config['training'].get('bf16', True)
        print_rank0(f"Mixed precision: {'BF16' if self.use_amp else 'FP32'}", self.rank)
        
        self._init_model()
        self._init_data()
        self._init_optimizer()
        
        self.log_interval = int(config['training'].get('log_interval', 10))
        self.save_interval = int(config['training'].get('save_interval', 1000))
        self.eval_interval = int(config['training'].get('eval_interval', 500))
    
    def _init_model(self):
        """Initialize model"""
        model_config = self.config['model']
        pretrained_model = model_config.get('pretrained_model')
        
        print_rank0(f"\nLoading model: {pretrained_model}", self.rank)
        
        config = AutoConfig.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            local_files_only=True
        )
        if config.pad_token_id is None:
            config.pad_token_id = 0
        
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = 0
        
        self.model = self.model.to(self.device)
        
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print_rank0(f"Total parameters: {total_params:,}", self.rank)
    
    def _init_data(self):
        """Initialize data loaders"""
        data_config = self.config['data']
        seq_length = int(data_config.get('seq_length', 65536))
        
        train_ds = ChickenGenomeDataset(
            data_path=data_config['train_path'],
            seq_length=seq_length,
            mlm=True,
            mlm_probability=float(data_config.get('mlm_probability', 0.15)),
            rc_aug=data_config.get('rc_aug', True),
        )
        
        val_ds = ChickenGenomeDataset(
            data_path=data_config['val_path'],
            seq_length=seq_length,
            mlm=True,
            mlm_probability=float(data_config.get('mlm_probability', 0.15)),
            rc_aug=False,
        )
        
        # Distributed sampler
        train_sampler = DistributedSampler(train_ds) if self.world_size > 1 else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.world_size > 1 else None
        
        batch_size = int(data_config.get('batch_size', 4))
        num_workers = int(data_config.get('num_workers', 4))
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        self.train_sampler = train_sampler
        
        print_rank0(f"Training batches: {len(self.train_loader)}", self.rank)
        print_rank0(f"Validation batches: {len(self.val_loader)}", self.rank)
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        opt_config = self.config.get('optimizer', {})
        train_config = self.config['training']
        
        lr = float(opt_config.get('lr', 1e-4))
        weight_decay = float(opt_config.get('weight_decay', 0.01))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        max_steps = int(train_config.get('max_steps', 10000))
        warmup_steps = int(train_config.get('warmup_steps', 500))
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        
        self.max_steps = max_steps
        self.grad_accum_steps = int(train_config.get('gradient_accumulation_steps', 1))
        self.max_grad_norm = float(train_config.get('max_grad_norm', 1.0))
        
        print_rank0(f"Learning rate: {lr}", self.rank)
        print_rank0(f"Max steps: {max_steps}", self.rank)
        print_rank0(f"Warmup steps: {warmup_steps}", self.rank)
        print_rank0(f"Gradient accumulation: {self.grad_accum_steps}", self.rank)
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
        
        avg_loss = total_loss / total_samples
        
        # Sync across processes
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        self.model.train()
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print_rank0("\n" + "="*60, self.rank)
        print_rank0("Starting training...", self.rank)
        print_rank0("="*60, self.rank)
        
        self.model.train()
        global_step = 0
        epoch = 0
        running_loss = 0
        
        # Tensorboard
        writer = None
        if HAS_TENSORBOARD and is_main_process(self.rank):
            writer = SummaryWriter(self.output_dir / 'logs')
        
        start_time = time.time()
        
        while global_step < self.max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast(dtype=torch.bfloat16):
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss / self.grad_accum_steps
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / self.grad_accum_steps
                
                # Backward pass
                loss.backward()
                running_loss += loss.item()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % self.log_interval == 0:
                        avg_loss = running_loss / self.log_interval
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        print_rank0(
                            f"Step {global_step}/{self.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {elapsed:.1f}s",
                            self.rank
                        )
                        
                        if writer:
                            writer.add_scalar('train/loss', avg_loss, global_step)
                            writer.add_scalar('train/lr', lr, global_step)
                        
                        running_loss = 0
                    
                    # Evaluation
                    if global_step % self.eval_interval == 0:
                        val_loss = self.evaluate()
                        print_rank0(f"Validation loss: {val_loss:.4f}", self.rank)
                        
                        if writer:
                            writer.add_scalar('val/loss', val_loss, global_step)
                    
                    # Save checkpoint
                    if global_step % self.save_interval == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            global_step, running_loss, self.config,
                            self.output_dir, self.rank
                        )
                    
                    if global_step >= self.max_steps:
                        break
            
            epoch += 1
        
        # Final save
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            global_step, running_loss, self.config,
            self.output_dir, self.rank
        )
        
        if writer:
            writer.close()
        
        print_rank0("\n" + "="*60, self.rank)
        print_rank0("Training complete!", self.rank)
        print_rank0("="*60, self.rank)


def main():
    parser = argparse.ArgumentParser(description='Train Chicken Caduceus')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    trainer = Trainer(config)
    trainer.train()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
