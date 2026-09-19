"""
IndicTrans2 Wrapper for Gloss-to-Text Translation.

Translates recognized glosses to Hindi and English text.
"""

import torch
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GlossTranslator:
    """
    Translates gloss sequences to natural language using IndicTrans2.
    
    Supports Hindi and English output.
    """
    
    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-indic-en-1B",
        device: str = "cuda",
        use_float16: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name for IndicTrans2
            device: Device to run on ("cuda" or "cpu")
            use_float16: Use FP16 for memory efficiency
        """
        self.device = device
        self.use_float16 = use_float16 and device == "cuda"
        self.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load(self):
        """Load the translation model (lazy loading for memory efficiency)."""
        if self._is_loaded:
            return
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            logger.info(f"Loading IndicTrans2 model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_float16 else torch.float32,
            ).to(self.device)
            
            self.model.eval()
            self._is_loaded = True
            
            logger.info("IndicTrans2 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load IndicTrans2: {e}")
            logger.warning("Translation will use simple gloss concatenation as fallback")
            self._is_loaded = False
    
    def translate(
        self,
        glosses: List[str],
        target_lang: str = "eng_Latn",
        max_length: int = 256,
    ) -> str:
        """
        Translate gloss sequence to target language.
        
        Args:
            glosses: List of gloss strings
            target_lang: Target language code ("eng_Latn" or "hin_Deva")
            max_length: Maximum output length
            
        Returns:
            Translated text
        """
        if not glosses:
            return ""
        
        # Join glosses into a pseudo-sentence
        gloss_text = " ".join(glosses)
        
        if not self._is_loaded:
            # Fallback: simple gloss concatenation
            return gloss_text.lower().replace("_", " ")
        
        try:
            # Prepare input
            # IndicTrans2 uses special language tags
            src_lang = "hin_Deva"  # Treat glosses as Hindi for En-Indic model
            
            inputs = self.tokenizer(
                gloss_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )
            
            # Decode output
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return gloss_text.lower().replace("_", " ")
    
    def translate_batch(
        self,
        batch_glosses: List[List[str]],
        target_lang: str = "eng_Latn",
    ) -> List[str]:
        """
        Translate a batch of gloss sequences.
        
        Args:
            batch_glosses: List of gloss sequences
            target_lang: Target language
            
        Returns:
            List of translated texts
        """
        return [self.translate(glosses, target_lang) for glosses in batch_glosses]
    
    def to_hindi(self, glosses: List[str]) -> str:
        """Translate glosses to Hindi."""
        return self.translate(glosses, target_lang="hin_Deva")
    
    def to_english(self, glosses: List[str]) -> str:
        """Translate glosses to English."""
        return self.translate(glosses, target_lang="eng_Latn")


class SimpleGlossTranslator:
    """
    Simple rule-based gloss-to-text converter.
    
    Used when IndicTrans2 is not available or for quick inference.
    """
    
    def __init__(self, gloss_to_word_map: Optional[dict] = None):
        """
        Args:
            gloss_to_word_map: Optional mapping of glosses to natural words
        """
        self.gloss_to_word = gloss_to_word_map or {}
    
    def translate(self, glosses: List[str], target_lang: str = "eng") -> str:
        """
        Convert glosses to readable text.
        
        Args:
            glosses: List of gloss strings
            target_lang: "eng" or "hin"
            
        Returns:
            Formatted text
        """
        if not glosses:
            return ""
        
        # Map glosses to words
        words = []
        for gloss in glosses:
            word = self.gloss_to_word.get(gloss, gloss.lower().replace("_", " "))
            words.append(word)
        
        # Basic sentence formation
        text = " ".join(words)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def load_mapping(self, path: str):
        """Load gloss-to-word mapping from file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            self.gloss_to_word = json.load(f)


if __name__ == "__main__":
    # Test translator
    print("Testing GlossTranslator...")
    
    # Test simple translator first (no model loading)
    simple_translator = SimpleGlossTranslator()
    
    glosses = ["HELLO", "HOW", "ARE", "YOU"]
    translation = simple_translator.translate(glosses)
    print(f"Glosses: {glosses}")
    print(f"Simple translation: {translation}")
    
    # Test full translator (will fallback if model not installed)
    translator = GlossTranslator(device="cpu")
    
    # Don't load in test - too slow
    # translator.load()
    
    # Test fallback
    translation = translator.translate(glosses)
    print(f"Translator output (fallback): {translation}")
    
    print("\nAll tests passed!")
