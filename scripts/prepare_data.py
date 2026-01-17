#!/usr/bin/env python
"""
Chicken Genome Data Preparation for PoultryCaduceus Pre-training

This script downloads and preprocesses the GRCg6a chicken genome
for Caduceus pre-training.

Usage:
    python scripts/prepare_data.py --output_dir data/pretrain
    
    # Custom parameters
    python scripts/prepare_data.py \
        --output_dir data/pretrain \
        --seq_length 65536 \
        --stride 32768 \
        --max_n_ratio 0.05

Based on: Chicken_Caduceus_Pretrain_GRCg6a_v2.ipynb
"""

import os
import gzip
import random
import json
import argparse
import urllib.request
import pickle
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from collections import defaultdict


# ============================================
# Default Configuration
# ============================================
DEFAULT_CONFIG = {
    # Sequence parameters
    'seq_length': 65536,           # 65k bp, Caduceus training sequence length
    'stride': 32768,               # 50% overlap
    'min_seq_length': 10000,       # Minimum sequence length
    
    # Quality filtering
    'max_n_ratio': 0.05,           # Maximum N ratio 5%
    'min_gc_ratio': 0.30,          # Minimum GC content
    'max_gc_ratio': 0.70,          # Maximum GC content
    
    # Dataset split
    'val_ratio': 0.02,             # Validation set ratio
    'test_ratio': 0.01,            # Test set ratio
    
    # Chromosome filtering (use main chromosomes only)
    'use_main_chromosomes': True,
    'main_chromosomes': [str(i) for i in range(1, 34)] + ['W', 'Z', 'MT'],
    
    # Random seed
    'seed': 42,
}

# Genome download URLs
GENOME_URL = 'https://ftp.ensembl.org/pub/release-104/fasta/gallus_gallus/dna/Gallus_gallus.GRCg6a.dna.toplevel.fa.gz'
GENOME_URL_BACKUP = 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/315/GCF_000002315.6_GRCg6a/GCF_000002315.6_GRCg6a_genomic.fna.gz'


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_genome(output_dir: Path) -> Path:
    """Download GRCg6a chicken genome."""
    genome_gz = output_dir / 'GRCg6a_genome.fa.gz'
    
    if genome_gz.exists():
        print(f"Genome file already exists: {genome_gz}")
        return genome_gz
    
    print("Downloading chicken genome GRCg6a (galGal6)...")
    print("Expected size: ~350 MB\n")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, desc="Ensembl GRCg6a") as t:
            urllib.request.urlretrieve(GENOME_URL, filename=genome_gz, reporthook=t.update_to)
        print(f"\nDownload complete! Size: {os.path.getsize(genome_gz)/1e6:.1f} MB")
    except Exception as e:
        print(f"Ensembl download failed: {e}")
        print("Trying NCBI backup...\n")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, desc="NCBI GRCg6a") as t:
                urllib.request.urlretrieve(GENOME_URL_BACKUP, filename=genome_gz, reporthook=t.update_to)
            print(f"\nDownload complete! Size: {os.path.getsize(genome_gz)/1e6:.1f} MB")
        except Exception as e2:
            raise RuntimeError(f"Both downloads failed: {e2}")
    
    return genome_gz


def decompress_genome(genome_gz: Path) -> Path:
    """Decompress genome file."""
    genome_fa = genome_gz.with_suffix('').with_suffix('.fa')
    
    if genome_fa.exists():
        print(f"Decompressed file already exists: {genome_fa}")
        return genome_fa
    
    print("Decompressing genome file...")
    with gzip.open(genome_gz, 'rb') as f_in:
        with open(genome_fa, 'wb') as f_out:
            total_size = os.path.getsize(genome_gz)
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Decompressing") as pbar:
                while True:
                    chunk = f_in.read(1024*1024)  # 1MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"\nDecompression complete! Size: {os.path.getsize(genome_fa)/1e9:.2f} GB")
    return genome_fa


def load_genome(genome_fa: Path, config: dict) -> dict:
    """Load and analyze genome."""
    print("Loading genome...")
    
    chromosomes = {}
    stats = defaultdict(int)
    
    for record in tqdm(SeqIO.parse(genome_fa, 'fasta'), desc="Reading chromosomes"):
        # Parse chromosome name
        chrom_id = record.id
        
        # Extract chromosome number/name
        if chrom_id.isdigit():
            chrom_name = chrom_id
        elif chrom_id.startswith('chr'):
            chrom_name = chrom_id[3:]
        elif '_' in chrom_id:
            # Handle names like "1_random" or scaffold names
            parts = chrom_id.split('_')
            chrom_name = parts[0]
        else:
            chrom_name = chrom_id
        
        # Filter to main chromosomes if configured
        if config['use_main_chromosomes']:
            if chrom_name not in config['main_chromosomes']:
                continue
        
        seq = str(record.seq).upper()
        chromosomes[chrom_name] = seq
        
        # Calculate statistics
        stats['total_length'] += len(seq)
        stats['gc_count'] += seq.count('G') + seq.count('C')
        stats['n_count'] += seq.count('N')
    
    # Print statistics
    print(f"\n{'='*50}")
    print("GRCg6a Genome Statistics")
    print('='*50)
    print(f"Chromosomes loaded: {len(chromosomes)}")
    print(f"Total length: {stats['total_length']:,} bp ({stats['total_length']/1e9:.2f} Gb)")
    print(f"GC content: {stats['gc_count']/stats['total_length']*100:.2f}%")
    print(f"N content: {stats['n_count']/stats['total_length']*100:.2f}%")
    
    print("\nChromosome lengths:")
    for chrom in sorted(chromosomes.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        print(f"  chr{chrom}: {len(chromosomes[chrom])/1e6:.1f} Mb")
    
    return chromosomes


def extract_sequences(chromosomes: dict, config: dict) -> list:
    """Extract training sequences using sliding window."""
    print(f"\nExtracting sequences (length={config['seq_length']}, stride={config['stride']})...")
    
    sequences = []
    stats = {
        'total_windows': 0,
        'passed': 0,
        'failed_n': 0,
        'failed_gc': 0,
        'failed_length': 0,
    }
    
    for chrom_name, chrom_seq in tqdm(chromosomes.items(), desc="Processing chromosomes"):
        chrom_len = len(chrom_seq)
        
        # Skip short chromosomes
        if chrom_len < config['min_seq_length']:
            continue
        
        # Sliding window
        for start in range(0, chrom_len - config['seq_length'] + 1, config['stride']):
            stats['total_windows'] += 1
            
            end = start + config['seq_length']
            seq = chrom_seq[start:end]
            
            # Quality filtering
            n_ratio = seq.count('N') / len(seq)
            if n_ratio > config['max_n_ratio']:
                stats['failed_n'] += 1
                continue
            
            gc_count = seq.count('G') + seq.count('C')
            gc_ratio = gc_count / (len(seq) - seq.count('N') + 1e-10)
            if gc_ratio < config['min_gc_ratio'] or gc_ratio > config['max_gc_ratio']:
                stats['failed_gc'] += 1
                continue
            
            stats['passed'] += 1
            sequences.append({
                'sequence': seq,
                'chromosome': chrom_name,
                'start': start,
                'end': end,
                'gc_ratio': gc_ratio,
                'n_ratio': n_ratio,
            })
        
        # Handle chromosome end (if remaining sequence is long enough)
        remaining = chrom_len % config['stride']
        if remaining >= config['min_seq_length']:
            start = chrom_len - config['seq_length']
            if start >= 0:
                seq = chrom_seq[start:]
                # Pad if necessary
                if len(seq) < config['seq_length']:
                    seq = seq + 'N' * (config['seq_length'] - len(seq))
                
                n_ratio = seq.count('N') / len(seq)
                if n_ratio <= config['max_n_ratio']:
                    gc_count = seq.count('G') + seq.count('C')
                    gc_ratio = gc_count / (len(seq) - seq.count('N') + 1e-10)
                    if config['min_gc_ratio'] <= gc_ratio <= config['max_gc_ratio']:
                        sequences.append({
                            'sequence': seq,
                            'chromosome': chrom_name,
                            'start': start,
                            'end': chrom_len,
                            'gc_ratio': gc_ratio,
                            'n_ratio': n_ratio,
                        })
    
    # Print statistics
    print(f"\n{'='*50}")
    print("Sequence Extraction Statistics")
    print('='*50)
    print(f"Total windows: {stats['total_windows']:,}")
    print(f"Passed: {stats['passed']:,} ({stats['passed']/stats['total_windows']*100:.1f}%)")
    print(f"Failed (N ratio): {stats['failed_n']:,}")
    print(f"Failed (GC ratio): {stats['failed_gc']:,}")
    
    return sequences


def split_dataset(sequences: list, config: dict) -> tuple:
    """Split sequences into train/val/test sets."""
    print("\nSplitting dataset...")
    
    random.seed(config['seed'])
    random.shuffle(sequences)
    
    n_total = len(sequences)
    n_val = int(n_total * config['val_ratio'])
    n_test = int(n_total * config['test_ratio'])
    n_train = n_total - n_val - n_test
    
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train + n_val]
    test_seqs = sequences[n_train + n_val:]
    
    print(f"Train: {len(train_seqs):,} sequences")
    print(f"Val: {len(val_seqs):,} sequences")
    print(f"Test: {len(test_seqs):,} sequences")
    
    return train_seqs, val_seqs, test_seqs


def save_sequences_hdf5(sequences: list, output_path: Path, config: dict):
    """Save sequences to HDF5 format for efficient loading."""
    print(f"\nSaving to {output_path}...")
    
    # Encode sequences
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    n_seqs = len(sequences)
    seq_length = config['seq_length']
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets
        seq_data = f.create_dataset(
            'sequences',
            shape=(n_seqs, seq_length),
            dtype=np.uint8,
            chunks=(min(1000, n_seqs), seq_length),
            compression='gzip',
            compression_opts=4
        )
        
        # Metadata
        chroms = f.create_dataset('chromosomes', shape=(n_seqs,), dtype=h5py.special_dtype(vlen=str))
        starts = f.create_dataset('starts', shape=(n_seqs,), dtype=np.int64)
        ends = f.create_dataset('ends', shape=(n_seqs,), dtype=np.int64)
        gc_ratios = f.create_dataset('gc_ratios', shape=(n_seqs,), dtype=np.float32)
        
        # Fill data
        for i, seq_info in enumerate(tqdm(sequences, desc="Encoding")):
            seq = seq_info['sequence']
            encoded = np.array([char_to_int.get(c, 4) for c in seq], dtype=np.uint8)
            seq_data[i] = encoded
            chroms[i] = seq_info['chromosome']
            starts[i] = seq_info['start']
            ends[i] = seq_info['end']
            gc_ratios[i] = seq_info['gc_ratio']
        
        # Save config
        f.attrs['config'] = json.dumps(config)
        f.attrs['n_sequences'] = n_seqs
        f.attrs['seq_length'] = seq_length
    
    print(f"Saved {n_seqs:,} sequences to {output_path}")
    print(f"File size: {os.path.getsize(output_path)/1e6:.1f} MB")


def save_sequences_text(sequences: list, output_path: Path):
    """Save sequences to text format (one per line)."""
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, 'w') as f:
        for seq_info in tqdm(sequences, desc="Writing"):
            f.write(seq_info['sequence'] + '\n')
    
    print(f"Saved {len(sequences):,} sequences")
    print(f"File size: {os.path.getsize(output_path)/1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare chicken genome data for PoultryCaduceus pre-training'
    )
    parser.add_argument('--output_dir', type=str, default='data/pretrain',
                        help='Output directory')
    parser.add_argument('--seq_length', type=int, default=65536,
                        help='Sequence length (default: 65536)')
    parser.add_argument('--stride', type=int, default=32768,
                        help='Sliding window stride (default: 32768)')
    parser.add_argument('--max_n_ratio', type=float, default=0.05,
                        help='Maximum N ratio (default: 0.05)')
    parser.add_argument('--val_ratio', type=float, default=0.02,
                        help='Validation set ratio (default: 0.02)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--format', type=str, choices=['hdf5', 'text', 'both'],
                        default='both', help='Output format')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip genome download (use existing file)')
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['seq_length'] = args.seq_length
    config['stride'] = args.stride
    config['max_n_ratio'] = args.max_n_ratio
    config['val_ratio'] = args.val_ratio
    config['seed'] = args.seed
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genome_dir = output_dir / 'genome'
    genome_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("PoultryCaduceus Data Preparation")
    print("="*60)
    print(f"\nSequence length: {config['seq_length']:,} bp")
    print(f"Stride: {config['stride']:,} bp")
    print(f"Max N ratio: {config['max_n_ratio']*100}%")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download genome
    if not args.skip_download:
        genome_gz = download_genome(genome_dir)
        genome_fa = decompress_genome(genome_gz)
    else:
        genome_fa = genome_dir / 'GRCg6a_genome.fa'
        if not genome_fa.exists():
            raise FileNotFoundError(f"Genome file not found: {genome_fa}")
    
    # Step 2: Load genome
    chromosomes = load_genome(genome_fa, config)
    
    # Step 3: Extract sequences
    sequences = extract_sequences(chromosomes, config)
    
    # Step 4: Split dataset
    train_seqs, val_seqs, test_seqs = split_dataset(sequences, config)
    
    # Step 5: Save sequences
    if args.format in ['hdf5', 'both']:
        save_sequences_hdf5(train_seqs, output_dir / 'train.h5', config)
        save_sequences_hdf5(val_seqs, output_dir / 'val.h5', config)
        if test_seqs:
            save_sequences_hdf5(test_seqs, output_dir / 'test.h5', config)
    
    if args.format in ['text', 'both']:
        save_sequences_text(train_seqs, output_dir / 'train.txt')
        save_sequences_text(val_seqs, output_dir / 'val.txt')
        if test_seqs:
            save_sequences_text(test_seqs, output_dir / 'test.txt')
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nOutput files in: {output_dir}")
    print("\nNext steps:")
    print("1. Transfer data to training server")
    print("2. Run: python scripts/pretrain.py --config configs/pretrain.yaml")


if __name__ == '__main__':
    main()
