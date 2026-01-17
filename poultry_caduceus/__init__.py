"""
PoultryCaduceus: A Bidirectional DNA Language Model for Chicken Genome

This package provides pre-trained DNA language models specifically designed
for the chicken (Gallus gallus) genome, with support for regulatory activity
prediction, eQTL analysis, and GWAS fine-mapping.

Example:
    >>> from poultry_caduceus import PoultryCaduceus
    >>> model = PoultryCaduceus.from_pretrained("poultry-caduceus-base")
    >>> embeddings = model.get_embeddings("ATGCGATCGATCG")
"""

from poultry_caduceus.__version__ import __version__
from poultry_caduceus.model import (
    PoultryCaduceus,
    PoultryCaduceusMPRA,
    PoultryCaduceusEQTL,
)
from poultry_caduceus.tokenizer import DNATokenizer
from poultry_caduceus.config import PoultryCaduceusConfig

__all__ = [
    "__version__",
    "PoultryCaduceus",
    "PoultryCaduceusMPRA",
    "PoultryCaduceusEQTL",
    "DNATokenizer",
    "PoultryCaduceusConfig",
]

# Package metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__url__ = "https://github.com/YOUR_USERNAME/PoultryCaduceus"
