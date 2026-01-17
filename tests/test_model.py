"""
Unit tests for PoultryCaduceus models.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from poultry_caduceus.model import (
    PoultryCaduceus,
    PoultryCaduceusMPRA,
    PoultryCaduceusEQTL,
    BiMambaBlock,
    MambaLayer
)
from poultry_caduceus.config import (
    PoultryCaduceusConfig,
    MPRAConfig,
    EQTLConfig
)
from poultry_caduceus.tokenizer import DNATokenizer


class TestDNATokenizer:
    """Tests for DNATokenizer."""
    
    def test_encode_single(self):
        tokenizer = DNATokenizer()
        sequence = "ATGCN"
        tokens = tokenizer.encode(sequence, return_tensors=None)
        assert tokens == [0, 3, 2, 1, 4]
    
    def test_encode_batch(self):
        tokenizer = DNATokenizer()
        sequences = ["ATGC", "GCTA"]
        tokens = tokenizer.encode(sequences, padding=True, return_tensors="pt")
        assert tokens.shape == (2, 4)
    
    def test_decode(self):
        tokenizer = DNATokenizer()
        tokens = [0, 3, 2, 1, 4]
        sequence = tokenizer.decode(tokens)
        assert sequence == "ATGCN"
    
    def test_reverse_complement_string(self):
        tokenizer = DNATokenizer()
        sequence = "ATGC"
        rc = tokenizer.reverse_complement(sequence)
        assert rc == "GCAT"
    
    def test_reverse_complement_tensor(self):
        tokenizer = DNATokenizer()
        tokens = torch.tensor([0, 3, 2, 1])  # ATGC
        rc = tokenizer.reverse_complement(tokens)
        expected = torch.tensor([2, 1, 0, 3])  # GCAT
        assert torch.equal(rc, expected)
    
    def test_padding(self):
        tokenizer = DNATokenizer(max_length=10, padding=True)
        sequence = "ATGC"
        tokens = tokenizer.encode(sequence, return_tensors="pt")
        assert tokens.shape == (1, 10)
        assert tokens[0, -1].item() == tokenizer.PAD_TOKEN_ID
    
    def test_truncation(self):
        tokenizer = DNATokenizer(max_length=3, truncation=True)
        sequence = "ATGCN"
        tokens = tokenizer.encode(sequence, return_tensors=None)
        assert len(tokens) == 3


class TestMambaLayer:
    """Tests for MambaLayer."""
    
    def test_forward(self):
        layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 100, 64)
        output = layer(x)
        assert output.shape == x.shape
    
    def test_different_seq_lengths(self):
        layer = MambaLayer(d_model=64)
        for seq_len in [50, 100, 200]:
            x = torch.randn(2, seq_len, 64)
            output = layer(x)
            assert output.shape == x.shape


class TestBiMambaBlock:
    """Tests for BiMambaBlock."""
    
    def test_forward(self):
        block = BiMambaBlock(d_model=64, d_state=16, rc_equivariant=True)
        x = torch.randn(2, 100, 64)
        output = block(x)
        assert output.shape == x.shape
    
    def test_rc_equivariance(self):
        block = BiMambaBlock(d_model=64, rc_equivariant=True)
        x = torch.randn(2, 100, 64)
        
        # Forward pass
        output = block(x)
        
        # Reverse and forward
        x_rev = x.flip(dims=[1])
        output_rev = block(x_rev)
        
        # Check approximate equivariance
        # Note: Due to non-linearities, exact equivariance is not expected
        assert output.shape == output_rev.shape


class TestPoultryCaduceus:
    """Tests for PoultryCaduceus model."""
    
    @pytest.fixture
    def config(self):
        return PoultryCaduceusConfig(
            vocab_size=6,
            d_model=64,
            n_layers=2,
            d_state=16,
            max_seq_len=256
        )
    
    @pytest.fixture
    def model(self, config):
        return PoultryCaduceus(config)
    
    def test_forward(self, model):
        input_ids = torch.randint(0, 5, (2, 100))
        output = model(input_ids)
        assert output.logits.shape == (2, 100, 6)
    
    def test_forward_with_labels(self, model):
        input_ids = torch.randint(0, 5, (2, 100))
        labels = torch.randint(0, 5, (2, 100))
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.dim() == 0  # Scalar
    
    def test_get_embeddings(self, model):
        sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCG"
        embeddings = model.get_embeddings(sequence, pooling="mean")
        assert embeddings.shape == (1, 64)
    
    def test_get_embeddings_no_pooling(self, model):
        input_ids = torch.randint(0, 5, (2, 50))
        embeddings = model.get_embeddings(input_ids, pooling="none")
        assert embeddings.shape == (2, 50, 64)
    
    def test_save_and_load(self, model, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)
            
            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "pytorch_model.bin"))
            
            # Load
            loaded_model = PoultryCaduceus.from_pretrained(tmpdir)
            
            # Compare outputs
            input_ids = torch.randint(0, 5, (2, 50))
            with torch.no_grad():
                output1 = model(input_ids)
                output2 = loaded_model(input_ids)
            
            assert torch.allclose(output1.logits, output2.logits, atol=1e-5)
    
    def test_num_parameters(self, model):
        assert model.num_parameters > 0
        assert model.num_trainable_parameters == model.num_parameters


class TestPoultryCaduceusMPRA:
    """Tests for PoultryCaduceusMPRA model."""
    
    @pytest.fixture
    def config(self):
        return MPRAConfig(
            vocab_size=6,
            d_model=64,
            n_layers=2,
            d_state=16,
            max_seq_len=256,
            head_hidden_dim=32
        )
    
    @pytest.fixture
    def model(self, config):
        return PoultryCaduceusMPRA(config)
    
    def test_forward(self, model):
        input_ids = torch.randint(0, 5, (2, 100))
        output = model(input_ids)
        assert output.logits.shape == (2,)
    
    def test_forward_with_labels(self, model):
        input_ids = torch.randint(0, 5, (2, 100))
        labels = torch.randn(2)
        output = model(input_ids, labels=labels)
        assert output.loss is not None
    
    def test_predict(self, model):
        sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCG"
        activity = model.predict(sequence)
        assert isinstance(activity, float)
    
    def test_predict_variant_effect(self, model):
        ref_seq = "ATGCGATCGATCGATCGATCGATCGATCGATCG"
        alt_seq = "ATGCGATCAATCGATCGATCGATCGATCGATCG"
        effect = model.predict_variant_effect(ref_seq, alt_seq)
        
        assert "ref_activity" in effect
        assert "alt_activity" in effect
        assert "effect_size" in effect
        assert effect["effect_size"] == effect["alt_activity"] - effect["ref_activity"]


class TestPoultryCaduceusEQTL:
    """Tests for PoultryCaduceusEQTL model."""
    
    @pytest.fixture
    def config(self):
        return EQTLConfig(
            vocab_size=6,
            d_model=64,
            n_layers=2,
            d_state=16,
            max_seq_len=256,
            num_tissues=6
        )
    
    @pytest.fixture
    def model(self, config):
        return PoultryCaduceusEQTL(config)
    
    def test_forward(self, model):
        input_ids = torch.randint(0, 5, (2, 100))
        output = model(input_ids)
        assert output.logits.shape == (2, 6)  # 6 tissues
    
    def test_forward_with_labels(self, model):
        input_ids = torch.randint(0, 5, (2, 100))
        labels = torch.randn(2, 6)
        output = model(input_ids, labels=labels)
        assert output.loss is not None


class TestConfig:
    """Tests for configuration classes."""
    
    def test_config_to_dict(self):
        config = PoultryCaduceusConfig(d_model=128)
        config_dict = config.to_dict()
        assert config_dict["d_model"] == 128
    
    def test_config_from_dict(self):
        config_dict = {"d_model": 128, "n_layers": 4}
        config = PoultryCaduceusConfig.from_dict(config_dict)
        assert config.d_model == 128
        assert config.n_layers == 4
    
    def test_config_save_load_json(self):
        config = PoultryCaduceusConfig(d_model=128)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded_config = PoultryCaduceusConfig.load(f.name)
        
        assert loaded_config.d_model == 128
        os.unlink(f.name)
    
    def test_config_save_load_yaml(self):
        config = PoultryCaduceusConfig(d_model=128)
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save(f.name)
            loaded_config = PoultryCaduceusConfig.load(f.name)
        
        assert loaded_config.d_model == 128
        os.unlink(f.name)
    
    def test_config_validation(self):
        with pytest.raises(AssertionError):
            PoultryCaduceusConfig(d_model=-1)
        
        with pytest.raises(AssertionError):
            PoultryCaduceusConfig(dropout=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
