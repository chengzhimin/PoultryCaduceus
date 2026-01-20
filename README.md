# 🐔 PoultryCaduceus

**A Bidirectional DNA Language Model for Chicken Genome**

PoultryCaduceus is the first DNA foundation model specifically pre-trained on the chicken (*Gallus gallus*) genome, based on the [Caduceus](https://github.com/kuleshov-group/caduceus) architecture.

## ✨ Features

- 🧬 **Chicken-specific**: Pre-trained on GRCg6a (~1.1 Gb) genome
- 🔄 **Bidirectional**: Mamba-based bidirectional sequence modeling
- ⚡ **RC Equivariance**: Built-in reverse complement equivariance
- 📏 **Long-range**: Supports 65,536 bp context

## 📊 Model Info

| Parameter | Value |
|-----------|-------|
| Base Model | caduceus-ph (4-layer) |
| Hidden Dim | 256 |
| Vocab Size | 16 |
| Sequence Length | 65,536 bp |
| Training Steps | 10,000 |
| Hardware | 4x H200 (80GB) |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/chengzhimin/PoultryCaduceus.git
cd PoultryCaduceus
source setup_env.sh
```

### Load Model

```python
from transformers import AutoModelForMaskedLM

# Load from HuggingFace
model = AutoModelForMaskedLM.from_pretrained(
    "jamie0315/PoultryCaduceus",
    subfolder="checkpoint-10000",
    trust_remote_code=True
)

# Or load from local checkpoint
model = AutoModelForMaskedLM.from_pretrained(
    "./checkpoint-10000",
    trust_remote_code=True
)
```

### Get Sequence Embeddings

```python
import torch

# DNA vocabulary
DNA_VOCAB = {'A': 7, 'C': 8, 'G': 9, 'T': 10, 'N': 5, '[MASK]': 4}

# Encode sequence
sequence = "ATGCGATCGATCGATCG"
input_ids = torch.tensor([[DNA_VOCAB.get(c, 5) for c in sequence]])

# Get embeddings
model.eval()
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1]  # (batch, seq_len, 256)
```

---

## 🔧 Training from Scratch

### Step 1: Setup Environment

```bash
# Create conda environment
conda create -n caduceus_env python=3.10
conda activate caduceus_env

# Install dependencies
pip install torch transformers h5py biopython pyyaml tensorboard

# Install Caduceus (requires CUDA)
pip install caduceus-dna
```

### Step 2: Download Base Model

Download pre-trained model from [Caduceus](https://github.com/kuleshov-group/caduceus):

```bash
git lfs install
git clone https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-4 ./caduceus-ph-model
```

### Step 3: Prepare Data

Run data preparation notebook on Google Colab (for servers without internet):

```bash
# Run on Colab
notebooks/data_preparation.ipynb

# Download generated data file
# chicken_pretrain_data_GRCg6a.tar.gz

# Upload to server and extract
tar -xzf chicken_pretrain_data_GRCg6a.tar.gz
```

Data directory structure (from HuggingFace):
```
PoultryCaduceus/
├── checkpoint-10000/                    # Model weights
│   ├── config.json
│   └── model.safetensors
└── chicken_pretrain_data_GRCg6a/        # Pre-training data
    ├── train_65k.h5                     # Training set (~58,000 sequences)
    └── val_65k.h5                       # Validation set (~1,200 sequences)
```

### Step 4: Start Training

```bash
# Single GPU
python scripts/train_chicken_caduceus_v8.py --config configs/chicken_caduceus_10k.yaml

# Multi-GPU (4x H200)
torchrun --nproc_per_node=4 scripts/train_chicken_caduceus_v8.py \
    --config configs/chicken_caduceus_10k.yaml
```

### Step 5: Training Output

```
outputs/chicken_caduceus_10k/
├── checkpoint-1000/
├── checkpoint-2000/
├── ...
└── checkpoint-10000/    # Final model
    ├── config.json
    ├── model.safetensors
    └── training_state.pt
```

---

## 📁 Repository Structure

```
PoultryCaduceus/
├── README.md
├── LICENSE
├── setup_env.sh                      # Environment setup
├── configs/
│   └── chicken_caduceus_10k.yaml     # Training config
├── scripts/
│   ├── chicken_dataset.py            # Dataset class
│   └── train_chicken_caduceus_v8.py  # Training script
└── notebooks/
    └── data_preparation.ipynb        # Data preparation (Colab)
```

---

## 📖 Training Configuration

```yaml
# chicken_caduceus_10k.yaml

model:
  pretrained_model: ./caduceus-ph-model  # Base model path

data:
  train_path: chicken_pretrain_data_GRCg6a/train_65k.h5
  val_path: chicken_pretrain_data_GRCg6a/val_65k.h5
  seq_length: 65536      # Sequence length
  batch_size: 6          # Batch size per GPU
  mlm_probability: 0.15  # Mask ratio
  rc_aug: true           # Reverse complement augmentation

training:
  max_steps: 10000       # Training steps
  warmup_steps: 500
  gradient_accumulation_steps: 2
  bf16: true             # Mixed precision

optimizer:
  lr: 2e-4
  weight_decay: 0.01
```

---

## 🎯 Applications

- **MPRA Prediction**: Predict regulatory sequence activity
- **eQTL Analysis**: Identify expression quantitative trait loci
- **GWAS Fine-mapping**: Prioritize causal variants
- **Regulatory Element Annotation**: Identify enhancers, promoters, etc.

---

## 📜 License

MIT License

## 🔗 Links

- 🤗 **HuggingFace**: [jamie0315/PoultryCaduceus](https://huggingface.co/jamie0315/PoultryCaduceus)
