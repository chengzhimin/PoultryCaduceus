#!/usr/bin/env python
"""
Upload PoultryCaduceus model to HuggingFace Hub.

Usage:
    # First login to HuggingFace
    huggingface-cli login
    
    # Then upload model
    python scripts/upload_to_huggingface.py \
        --model_path checkpoints/best_model.pt \
        --repo_name poultry-caduceus-base \
        --username YOUR_USERNAME

    # Upload MPRA fine-tuned model
    python scripts/upload_to_huggingface.py \
        --model_path checkpoints/mpra/best_model.pt \
        --repo_name poultry-caduceus-mpra \
        --username YOUR_USERNAME \
        --model_type mpra
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
from pathlib import Path

try:
    from huggingface_hub import (
        HfApi, 
        create_repo, 
        upload_folder,
        login,
        whoami
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")


def check_login():
    """Check if user is logged in to HuggingFace."""
    try:
        user_info = whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return user_info['name']
    except Exception as e:
        print(f"✗ Not logged in to HuggingFace")
        print("\nPlease login first:")
        print("  huggingface-cli login")
        print("  # or")
        print("  python -c \"from huggingface_hub import login; login()\"")
        return None


def create_model_card(args, model_info: dict) -> str:
    """Create README.md (Model Card) for HuggingFace."""
    
    model_card = f"""---
language:
- en
license: mit
tags:
- genomics
- dna
- chicken
- poultry
- caduceus
- biology
- bioinformatics
datasets:
- custom
metrics:
- pearson_r
- accuracy
library_name: transformers
pipeline_tag: feature-extraction
---

# PoultryCaduceus {'MPRA' if args.model_type == 'mpra' else 'Base'}

A bidirectional DNA language model pre-trained on the chicken (*Gallus gallus*) genome.

## Model Description

**PoultryCaduceus** is the first DNA foundation model specifically designed for the chicken genome. Built upon the [Caduceus](https://github.com/kuleshov-group/caduceus) architecture, it features:

- 🧬 **Chicken-specific pre-training** on GRCg6a reference genome (~1.1 Gb)
- 🔄 **Bidirectional Mamba** architecture with reverse complement equivariance  
- 📏 **Long-range modeling** up to 65,536 bp context
- 🧪 **{'MPRA fine-tuning for regulatory activity prediction' if args.model_type == 'mpra' else 'Self-supervised pre-training with masked language modeling'}**

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Caduceus (BiMamba) |
| Hidden dimension | {model_info.get('d_model', 256)} |
| Layers | {model_info.get('n_layers', 16)} |
| Max sequence length | {model_info.get('max_seq_len', 65536)} |
| Vocabulary size | {model_info.get('vocab_size', 12)} |
| Parameters | {model_info.get('num_params', 'N/A')} |

## Training Data

- **Genome**: GRCg6a (galGal6) - Chicken reference genome
- **Source**: Ensembl Release 104
- **Total length**: ~1.1 Gb
- **Chromosomes**: 1-33, W, Z, MT
- **Training sequences**: ~60,000 (65k bp each)

## Usage

### Installation

```bash
pip install poultry-caduceus
# or
git clone https://github.com/{args.username}/PoultryCaduceus.git
cd PoultryCaduceus
pip install -e .
```

### Quick Start

```python
from poultry_caduceus import PoultryCaduceus{', PoultryCaduceusMPRA' if args.model_type == 'mpra' else ''}

# Load model
model = {'PoultryCaduceusMPRA' if args.model_type == 'mpra' else 'PoultryCaduceus'}.from_pretrained("{args.username}/{args.repo_name}")

# Get embeddings
sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCG"
{'activity = model.predict(sequence)' if args.model_type == 'mpra' else 'embeddings = model.get_embeddings(sequence)'}
```

### Using with Transformers

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("{args.username}/{args.repo_name}", trust_remote_code=True)

# Tokenize sequence
sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCG"
# ... (see full documentation)
```

## Performance

{'### MPRA Prediction' if args.model_type == 'mpra' else '### Pre-training'}

| Metric | Value |
|--------|-------|
| {'Pearson r' if args.model_type == 'mpra' else 'MLM Accuracy'} | X.XX |
| {'Spearman ρ' if args.model_type == 'mpra' else 'Perplexity'} | X.XX |

## Citation

```bibtex
@article{{poultrycaduceus2024,
  title={{PoultryCaduceus: A Bidirectional DNA Language Model for Chicken Genome}},
  author={{Your Name}},
  year={{2024}}
}}
```

## License

MIT License

## Acknowledgments

- [Caduceus](https://github.com/kuleshov-group/caduceus) for the base architecture
- Ensembl for the chicken reference genome
"""
    
    return model_card


def create_config_json(model_info: dict) -> dict:
    """Create config.json for HuggingFace."""
    
    config = {
        "architectures": ["PoultryCaduceus"],
        "model_type": "poultry_caduceus",
        "vocab_size": model_info.get('vocab_size', 12),
        "d_model": model_info.get('d_model', 256),
        "n_layers": model_info.get('n_layers', 16),
        "d_state": model_info.get('d_state', 64),
        "d_conv": model_info.get('d_conv', 4),
        "expand": model_info.get('expand', 2),
        "max_seq_len": model_info.get('max_seq_len', 65536),
        "rc_equivariant": model_info.get('rc_equivariant', True),
        "dropout": model_info.get('dropout', 0.1),
        "torch_dtype": "float32",
        "transformers_version": "4.30.0"
    }
    
    return config


def prepare_upload_folder(args, model_info: dict) -> Path:
    """Prepare folder structure for upload."""
    
    # Create temporary directory
    upload_dir = Path(tempfile.mkdtemp())
    
    print(f"\nPreparing upload folder: {upload_dir}")
    
    # 1. Copy model weights
    model_path = Path(args.model_path)
    if model_path.is_file():
        # Single file
        shutil.copy(model_path, upload_dir / "pytorch_model.bin")
        print(f"  ✓ Copied model weights")
    elif model_path.is_dir():
        # Directory with multiple files
        for f in model_path.glob("*.bin"):
            shutil.copy(f, upload_dir / f.name)
        for f in model_path.glob("*.pt"):
            shutil.copy(f, upload_dir / f.name)
        for f in model_path.glob("*.safetensors"):
            shutil.copy(f, upload_dir / f.name)
        print(f"  ✓ Copied model files from directory")
    
    # 2. Create config.json
    config = create_config_json(model_info)
    with open(upload_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Created config.json")
    
    # 3. Create README.md (Model Card)
    model_card = create_model_card(args, model_info)
    with open(upload_dir / "README.md", 'w') as f:
        f.write(model_card)
    print(f"  ✓ Created README.md (Model Card)")
    
    # 4. Copy tokenizer files if available
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
    for tf in tokenizer_files:
        tf_path = model_path.parent / tf if model_path.is_file() else model_path / tf
        if tf_path.exists():
            shutil.copy(tf_path, upload_dir / tf)
            print(f"  ✓ Copied {tf}")
    
    # 5. Create simple tokenizer config if not exists
    if not (upload_dir / "tokenizer_config.json").exists():
        tokenizer_config = {
            "tokenizer_class": "DNATokenizer",
            "vocab_size": model_info.get('vocab_size', 12),
            "model_max_length": model_info.get('max_seq_len', 65536),
        }
        with open(upload_dir / "tokenizer_config.json", 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"  ✓ Created tokenizer_config.json")
    
    return upload_dir


def upload_to_hub(args, upload_dir: Path):
    """Upload model to HuggingFace Hub."""
    
    api = HfApi()
    repo_id = f"{args.username}/{args.repo_name}"
    
    # Create repository
    print(f"\nCreating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True
        )
        print(f"  ✓ Repository created/exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ✗ Failed to create repository: {e}")
        return False
    
    # Upload files
    print(f"\nUploading files...")
    try:
        upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload PoultryCaduceus {args.model_type} model"
        )
        print(f"  ✓ Upload complete!")
        print(f"\n🎉 Model available at: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        return False


def extract_model_info(model_path: Path) -> dict:
    """Extract model information from checkpoint."""
    
    import torch
    
    model_info = {
        'd_model': 256,
        'n_layers': 16,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'max_seq_len': 65536,
        'vocab_size': 12,
        'rc_equivariant': True,
        'dropout': 0.1,
    }
    
    try:
        if model_path.is_file():
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path / 'pytorch_model.bin', map_location='cpu')
        
        # Extract config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                model_info.update(config.get('model', config))
        
        # Count parameters
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        num_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        model_info['num_params'] = f"{num_params:,}"
        
        print(f"\nExtracted model info:")
        print(f"  Parameters: {model_info['num_params']}")
        print(f"  d_model: {model_info['d_model']}")
        print(f"  n_layers: {model_info['n_layers']}")
        
    except Exception as e:
        print(f"Warning: Could not extract model info: {e}")
        print("Using default values...")
    
    return model_info


def main():
    parser = argparse.ArgumentParser(description='Upload PoultryCaduceus to HuggingFace Hub')
    
    parser.add_argument('--model_path', type=Path, required=True,
                        help='Path to model checkpoint (.pt file or directory)')
    parser.add_argument('--repo_name', type=str, required=True,
                        help='Repository name (e.g., poultry-caduceus-base)')
    parser.add_argument('--username', type=str, default=None,
                        help='HuggingFace username (auto-detected if logged in)')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'mpra', 'eqtl'],
                        help='Model type for documentation')
    parser.add_argument('--private', action='store_true',
                        help='Make repository private')
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("Error: huggingface_hub is required")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    # Check login
    username = check_login()
    if username is None:
        sys.exit(1)
    
    if args.username is None:
        args.username = username
    
    # Check model path
    if not args.model_path.exists():
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("PoultryCaduceus HuggingFace Upload")
    print("="*60)
    print(f"\nModel path: {args.model_path}")
    print(f"Repository: {args.username}/{args.repo_name}")
    print(f"Model type: {args.model_type}")
    print(f"Private: {args.private}")
    
    # Extract model info
    model_info = extract_model_info(args.model_path)
    
    # Prepare upload folder
    upload_dir = prepare_upload_folder(args, model_info)
    
    # Upload
    success = upload_to_hub(args, upload_dir)
    
    # Cleanup
    shutil.rmtree(upload_dir)
    
    if success:
        print("\n" + "="*60)
        print("Next steps:")
        print("="*60)
        print(f"1. Visit https://huggingface.co/{args.username}/{args.repo_name}")
        print("2. Edit the Model Card to add performance metrics")
        print("3. Add tags and update documentation")
        print("\nTo use the model:")
        print(f'  model = PoultryCaduceus.from_pretrained("{args.username}/{args.repo_name}")')
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
