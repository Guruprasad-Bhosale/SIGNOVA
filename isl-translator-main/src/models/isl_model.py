"""
Complete ISL Translator Model.

End-to-end model combining encoder, CTC decoder, and translator.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from .transformer import SignTransformer, HybridEncoder
from .stgcn import STGCN
from .ctc_decoder import CTCDecoder, CTCHead
from .translator import GlossTranslator, SimpleGlossTranslator

logger = logging.getLogger(__name__)


class ISLTranslator(nn.Module):
    """
    End-to-end Indian Sign Language Translator.
    
    Pipeline:
    1. Encode keypoints with ST-GCN/Transformer
    2. Decode to glosses with CTC
    3. Translate glosses to Hindi/English
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        encoder_type: str = "transformer",
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        stgcn_hidden: int = 64,
        stgcn_layers: int = 2,
        dropout: float = 0.1,
        use_translation: bool = True,
        translation_device: str = "cuda",
    ):
        """
        Args:
            vocab: Gloss vocabulary {gloss: index}
            encoder_type: "transformer", "stgcn", or "hybrid"
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            stgcn_hidden: ST-GCN hidden dimension
            stgcn_layers: Number of ST-GCN layers
            dropout: Dropout rate
            use_translation: Whether to use IndicTrans2
            translation_device: Device for translation model
        """
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_to_gloss = {v: k for k, v in vocab.items()}
        self.encoder_type = encoder_type
        
        # Build encoder
        if encoder_type == "transformer":
            self.encoder = SignTransformer(
                num_keypoints=543,
                keypoint_dim=3,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
            )
            encoder_dim = d_model
        elif encoder_type == "stgcn":
            self.encoder = STGCN(
                in_channels=3,
                hidden_channels=stgcn_hidden,
                out_channels=d_model,
                num_layers=stgcn_layers,
                dropout=dropout,
            )
            encoder_dim = d_model
        elif encoder_type == "hybrid":
            self.encoder = HybridEncoder(
                num_keypoints=543,
                keypoint_dim=3,
                stgcn_hidden=stgcn_hidden,
                stgcn_layers=stgcn_layers,
                d_model=d_model,
                nhead=nhead,
                num_transformer_layers=num_encoder_layers,
                dropout=dropout,
            )
            encoder_dim = d_model
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # CTC decoder
        self.ctc = CTCDecoder(
            input_dim=encoder_dim,
            vocab_size=self.vocab_size,
            hidden_dim=encoder_dim,
            dropout=dropout,
        )
        
        # Translator (lazy loaded)
        self.use_translation = use_translation
        self._translator = None
        self._translation_device = translation_device
        
        logger.info(f"Initialized ISLTranslator with {encoder_type} encoder")
        logger.info(f"Vocabulary size: {self.vocab_size}")
    
    @property
    def translator(self) -> GlossTranslator:
        """Lazy load translator."""
        if self._translator is None:
            if self.use_translation:
                self._translator = GlossTranslator(device=self._translation_device)
            else:
                self._translator = SimpleGlossTranslator()
        return self._translator
    
    def forward(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            keypoints: Input keypoints (N, T, V, C)
            lengths: Sequence lengths (N,)
            targets: Target gloss indices (N, S) for training
            target_lengths: Target lengths (N,)
            
        Returns:
            Dictionary with logits, loss (if training), etc.
        """
        # Encode
        features, enc_lengths = self.encoder(keypoints, lengths)
        
        # Get CTC logits
        logits = self.ctc(features, enc_lengths)
        
        output = {
            "logits": logits,
            "lengths": enc_lengths,
        }
        
        # Compute loss if targets provided
        if targets is not None and target_lengths is not None:
            loss = self.ctc.compute_loss(logits, targets, enc_lengths, target_lengths)
            output["loss"] = loss
        
        return output
    
    def decode(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        beam_width: int = 5,
    ) -> List[List[str]]:
        """
        Decode keypoints to gloss sequences.
        
        Args:
            keypoints: (N, T, V, C)
            lengths: (N,)
            beam_width: Beam search width (1 = greedy)
            
        Returns:
            List of gloss sequences
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(keypoints, lengths)
            logits = output["logits"]
            enc_lengths = output["lengths"]
            
            if beam_width <= 1:
                decoded_indices = self.ctc.decode_greedy(logits, enc_lengths)
            else:
                decoded_indices = self.ctc.decode_beam(logits, enc_lengths, beam_width)
        
        # Convert indices to glosses
        decoded_glosses = []
        for indices in decoded_indices:
            glosses = [self.idx_to_gloss.get(i, "<unk>") for i in indices]
            # Remove special tokens
            glosses = [g for g in glosses if g not in ["<blank>", "<sos>", "<eos>", "<unk>"]]
            decoded_glosses.append(glosses)
        
        return decoded_glosses
    
    def translate(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        target_lang: str = "eng_Latn",
        beam_width: int = 5,
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Full translation pipeline: keypoints → glosses → text.
        
        Args:
            keypoints: (N, T, V, C)
            lengths: (N,)
            target_lang: "eng_Latn" or "hin_Deva"
            beam_width: Beam width for CTC decoding
            
        Returns:
            glosses: List of gloss sequences
            translations: List of translated texts
        """
        # Decode to glosses
        glosses = self.decode(keypoints, lengths, beam_width)
        
        # Translate to text
        translations = []
        for gloss_seq in glosses:
            if isinstance(self.translator, GlossTranslator):
                text = self.translator.translate(gloss_seq, target_lang)
            else:
                text = self.translator.translate(gloss_seq)
            translations.append(text)
        
        return glosses, translations
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str, save_vocab: bool = True):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "encoder_type": self.encoder_type,
            "vocab_size": self.vocab_size,
        }
        if save_vocab:
            checkpoint["vocab"] = self.vocab
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, vocab: Optional[Dict[str, int]] = None, **kwargs):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        
        if vocab is None:
            vocab = checkpoint.get("vocab")
            if vocab is None:
                raise ValueError("Vocabulary not found in checkpoint and not provided")
        
        model = cls(
            vocab=vocab,
            encoder_type=checkpoint.get("encoder_type", "transformer"),
            **kwargs,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Model loaded from {path}")
        return model


def create_model(
    vocab: Dict[str, int],
    config: dict,
) -> ISLTranslator:
    """
    Create model from config dictionary.
    
    Args:
        vocab: Gloss vocabulary
        config: Model configuration
        
    Returns:
        ISLTranslator instance
    """
    model_config = config.get("model", {})
    
    return ISLTranslator(
        vocab=vocab,
        encoder_type=model_config.get("type", "transformer"),
        d_model=model_config.get("transformer", {}).get("d_model", 256),
        nhead=model_config.get("transformer", {}).get("nhead", 8),
        num_encoder_layers=model_config.get("transformer", {}).get("num_encoder_layers", 4),
        stgcn_hidden=model_config.get("stgcn", {}).get("hidden_channels", 64),
        stgcn_layers=model_config.get("stgcn", {}).get("num_layers", 4),
        dropout=model_config.get("transformer", {}).get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Test ISLTranslator
    print("Testing ISLTranslator...")
    
    # Create dummy vocab
    vocab = {"<blank>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for i in range(100):
        vocab[f"GLOSS_{i}"] = i + 4
    
    # Create model
    model = ISLTranslator(
        vocab=vocab,
        encoder_type="transformer",
        d_model=128,
        num_encoder_layers=2,
        use_translation=False,
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 50
    keypoints = torch.randn(batch_size, seq_len, 543, 3)
    lengths = torch.tensor([50, 40])
    targets = torch.randint(4, 50, (batch_size, 10))
    target_lengths = torch.tensor([10, 8])
    
    output = model(keypoints, lengths, targets, target_lengths)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test decoding
    glosses = model.decode(keypoints, lengths)
    print(f"Decoded glosses: {glosses}")
    
    # Test translation
    glosses, translations = model.translate(keypoints, lengths)
    print(f"Translations: {translations}")
    
    print("\nAll tests passed!")
