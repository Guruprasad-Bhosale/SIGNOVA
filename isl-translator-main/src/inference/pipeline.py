"""
Inference Pipeline for ISL Translation.

Provides easy-to-use interface for translating sign language videos.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

from ..preprocessing.keypoint_extractor import KeypointExtractor
from ..models.isl_model import ISLTranslator

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline for ISL translation.
    
    Usage:
        pipeline = InferencePipeline("checkpoints/best.pt")
        glosses, hindi, english = pipeline.translate_video("video.mp4")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        beam_width: int = 5,
        use_translation: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on
            beam_width: Beam width for CTC decoding
            use_translation: Whether to load IndicTrans2 for text translation
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.beam_width = beam_width
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = ISLTranslator.load(
            checkpoint_path,
            use_translation=use_translation,
            translation_device=self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize extractor
        self.extractor = KeypointExtractor()
        
        logger.info("Pipeline ready")
    
    def translate_video(
        self,
        video_path: str,
        return_keypoints: bool = False,
    ) -> Union[Tuple[List[str], str, str], Tuple[List[str], str, str, np.ndarray]]:
        """
        Translate a sign language video.
        
        Args:
            video_path: Path to video file
            return_keypoints: Whether to return extracted keypoints
            
        Returns:
            glosses: List of recognized glosses
            hindi: Hindi translation
            english: English translation
            keypoints: (optional) Extracted keypoints array
        """
        # Extract keypoints
        logger.info(f"Processing video: {video_path}")
        keypoints, metadata = self.extractor.extract_video(video_path, show_progress=True)
        
        # Interpolate missing
        for component in ["pose", "left_hand", "right_hand"]:
            keypoints = self.extractor.interpolate_missing(keypoints, metadata, component)
        
        # Prepare tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        keypoints_tensor = keypoints_tensor.unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(keypoints)]).to(self.device)
        
        # Translate
        with torch.no_grad():
            gloss_seqs, translations = self.model.translate(
                keypoints_tensor,
                lengths,
                target_lang="eng_Latn",
                beam_width=self.beam_width,
            )
        
        glosses = gloss_seqs[0]
        english = translations[0]
        
        # Get Hindi translation
        with torch.no_grad():
            _, hindi_translations = self.model.translate(
                keypoints_tensor,
                lengths,
                target_lang="hin_Deva",
                beam_width=self.beam_width,
            )
        hindi = hindi_translations[0]
        
        if return_keypoints:
            return glosses, hindi, english, keypoints
        return glosses, hindi, english
    
    def translate_keypoints(
        self,
        keypoints: np.ndarray,
    ) -> Tuple[List[str], str, str]:
        """
        Translate from pre-extracted keypoints.
        
        Args:
            keypoints: Keypoints array of shape (T, 543, 3)
            
        Returns:
            glosses, hindi, english
        """
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        keypoints_tensor = keypoints_tensor.unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(keypoints)]).to(self.device)
        
        with torch.no_grad():
            gloss_seqs, translations = self.model.translate(
                keypoints_tensor,
                lengths,
                beam_width=self.beam_width,
            )
        
        return gloss_seqs[0], "", translations[0]
    
    def batch_translate(
        self,
        video_paths: List[str],
    ) -> List[Tuple[List[str], str, str]]:
        """
        Translate multiple videos.
        
        Args:
            video_paths: List of video paths
            
        Returns:
            List of (glosses, hindi, english) tuples
        """
        results = []
        for path in video_paths:
            try:
                result = self.translate_video(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append(([], "", ""))
        return results
    
    def close(self):
        """Release resources."""
        self.extractor.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    print("InferencePipeline - use via:")
    print("  pipeline = InferencePipeline('checkpoints/best.pt')")
    print("  glosses, hindi, english = pipeline.translate_video('video.mp4')")
