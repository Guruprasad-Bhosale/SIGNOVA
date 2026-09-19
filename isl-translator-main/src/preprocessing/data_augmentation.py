"""
Data Augmentation Pipeline for Sign Language Keypoints.

Applies temporal and spatial augmentations to improve model robustness.
"""

import numpy as np
from typing import Tuple, Optional, List
import random


class DataAugmenter:
    """
    Augmentation pipeline for keypoint sequences.
    
    Supports:
    - Temporal: Speed perturbation, random cropping
    - Spatial: Rotation, scaling, translation
    - Noise: Gaussian jitter, landmark dropout
    """
    
    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: Tuple[float, float] = (-0.1, 0.1),
        gaussian_noise_std: float = 0.01,
        landmark_dropout_prob: float = 0.1,
        random_crop: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize augmenter with augmentation parameters.
        
        Args:
            speed_range: Range for temporal speed perturbation (min, max)
            rotation_range: Range for rotation in degrees
            scale_range: Range for scaling factor
            translation_range: Range for translation (normalized coordinates)
            gaussian_noise_std: Standard deviation for Gaussian noise
            landmark_dropout_prob: Probability of dropping each landmark
            random_crop: Whether to randomly crop temporal sequence
            seed: Random seed for reproducibility
        """
        self.speed_range = speed_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.gaussian_noise_std = gaussian_noise_std
        self.landmark_dropout_prob = landmark_dropout_prob
        self.random_crop = random_crop
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def speed_perturbation(
        self,
        keypoints: np.ndarray,
        speed_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply temporal speed perturbation by resampling frames.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3)
            speed_factor: Speed factor (< 1 = slower, > 1 = faster)
            
        Returns:
            Resampled keypoints
        """
        if speed_factor is None:
            speed_factor = np.random.uniform(*self.speed_range)
        
        T = len(keypoints)
        new_T = int(T / speed_factor)
        
        if new_T < 2:
            return keypoints
        
        # Linear interpolation for resampling
        old_indices = np.arange(T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        # Interpolate each landmark coordinate
        new_keypoints = np.zeros((new_T, keypoints.shape[1], keypoints.shape[2]), dtype=np.float32)
        
        for i in range(keypoints.shape[1]):
            for j in range(keypoints.shape[2]):
                new_keypoints[:, i, j] = np.interp(new_indices, old_indices, keypoints[:, i, j])
        
        return new_keypoints
    
    def temporal_crop(
        self,
        keypoints: np.ndarray,
        target_length: Optional[int] = None,
        min_ratio: float = 0.7,
    ) -> np.ndarray:
        """
        Randomly crop a temporal segment.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3)
            target_length: Target length (None = random crop ratio)
            min_ratio: Minimum ratio of sequence to keep
            
        Returns:
            Cropped keypoints
        """
        T = len(keypoints)
        
        if target_length is None:
            crop_ratio = np.random.uniform(min_ratio, 1.0)
            target_length = max(2, int(T * crop_ratio))
        
        if target_length >= T:
            return keypoints
        
        start_idx = np.random.randint(0, T - target_length + 1)
        return keypoints[start_idx:start_idx + target_length]
    
    def rotate_2d(
        self,
        keypoints: np.ndarray,
        angle_deg: Optional[float] = None,
        center: Tuple[float, float] = (0.5, 0.5),
    ) -> np.ndarray:
        """
        Apply 2D rotation around a center point.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3) or (num_landmarks, 3)
            angle_deg: Rotation angle in degrees
            center: Center of rotation (normalized coordinates)
            
        Returns:
            Rotated keypoints
        """
        if angle_deg is None:
            angle_deg = np.random.uniform(*self.rotation_range)
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        keypoints = keypoints.copy()
        
        # Apply rotation to x, y coordinates
        x = keypoints[..., 0] - center[0]
        y = keypoints[..., 1] - center[1]
        
        keypoints[..., 0] = cos_a * x - sin_a * y + center[0]
        keypoints[..., 1] = sin_a * x + cos_a * y + center[1]
        
        return keypoints
    
    def scale(
        self,
        keypoints: np.ndarray,
        scale_factor: Optional[float] = None,
        center: Tuple[float, float] = (0.5, 0.5),
    ) -> np.ndarray:
        """
        Apply scaling around a center point.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3) or (num_landmarks, 3)
            scale_factor: Scale factor
            center: Center of scaling
            
        Returns:
            Scaled keypoints
        """
        if scale_factor is None:
            scale_factor = np.random.uniform(*self.scale_range)
        
        keypoints = keypoints.copy()
        
        keypoints[..., 0] = (keypoints[..., 0] - center[0]) * scale_factor + center[0]
        keypoints[..., 1] = (keypoints[..., 1] - center[1]) * scale_factor + center[1]
        
        return keypoints
    
    def translate(
        self,
        keypoints: np.ndarray,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply translation to keypoints.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3) or (num_landmarks, 3)
            dx: X translation (normalized)
            dy: Y translation (normalized)
            
        Returns:
            Translated keypoints
        """
        if dx is None:
            dx = np.random.uniform(*self.translation_range)
        if dy is None:
            dy = np.random.uniform(*self.translation_range)
        
        keypoints = keypoints.copy()
        keypoints[..., 0] += dx
        keypoints[..., 1] += dy
        
        return keypoints
    
    def add_gaussian_noise(
        self,
        keypoints: np.ndarray,
        std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Add Gaussian noise to keypoint coordinates.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3)
            std: Standard deviation of noise
            
        Returns:
            Noisy keypoints
        """
        if std is None:
            std = self.gaussian_noise_std
        
        keypoints = keypoints.copy()
        noise = np.random.normal(0, std, keypoints[..., :2].shape)
        keypoints[..., :2] += noise
        
        return keypoints
    
    def landmark_dropout(
        self,
        keypoints: np.ndarray,
        dropout_prob: Optional[float] = None,
    ) -> np.ndarray:
        """
        Randomly zero out landmarks to simulate occlusion.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3)
            dropout_prob: Probability of dropping each landmark
            
        Returns:
            Keypoints with some landmarks zeroed
        """
        if dropout_prob is None:
            dropout_prob = self.landmark_dropout_prob
        
        keypoints = keypoints.copy()
        
        # Create dropout mask
        mask = np.random.random(keypoints.shape[:2]) > dropout_prob
        mask = mask[..., np.newaxis]
        
        keypoints = keypoints * mask
        
        return keypoints
    
    def __call__(
        self,
        keypoints: np.ndarray,
        augmentations: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Apply a sequence of augmentations.
        
        Args:
            keypoints: Shape (T, num_landmarks, 3)
            augmentations: List of augmentation names to apply
                Options: ["speed", "crop", "rotate", "scale", "translate", "noise", "dropout"]
                If None, applies all with random probability
                
        Returns:
            Augmented keypoints
        """
        if augmentations is None:
            # Apply each augmentation with 50% probability
            augmentations = []
            for aug in ["speed", "crop", "rotate", "scale", "translate", "noise", "dropout"]:
                if random.random() > 0.5:
                    augmentations.append(aug)
        
        for aug in augmentations:
            if aug == "speed":
                keypoints = self.speed_perturbation(keypoints)
            elif aug == "crop" and self.random_crop:
                keypoints = self.temporal_crop(keypoints)
            elif aug == "rotate":
                keypoints = self.rotate_2d(keypoints)
            elif aug == "scale":
                keypoints = self.scale(keypoints)
            elif aug == "translate":
                keypoints = self.translate(keypoints)
            elif aug == "noise":
                keypoints = self.add_gaussian_noise(keypoints)
            elif aug == "dropout":
                keypoints = self.landmark_dropout(keypoints)
        
        return keypoints
    
    def augment_batch(
        self,
        batch: List[np.ndarray],
        augmentations: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Apply augmentations to a batch of sequences.
        
        Args:
            batch: List of keypoint arrays
            augmentations: Augmentations to apply
            
        Returns:
            List of augmented keypoint arrays
        """
        return [self(kp, augmentations) for kp in batch]


if __name__ == "__main__":
    # Test augmentations
    print("Testing DataAugmenter...")
    
    # Create dummy keypoints (T=100 frames, 543 landmarks, 3 coords)
    dummy_kp = np.random.rand(100, 543, 3).astype(np.float32)
    
    augmenter = DataAugmenter(seed=42)
    
    # Test individual augmentations
    print(f"Original shape: {dummy_kp.shape}")
    
    aug_kp = augmenter.speed_perturbation(dummy_kp, speed_factor=1.2)
    print(f"After speed (1.2x): {aug_kp.shape}")
    
    aug_kp = augmenter.temporal_crop(dummy_kp, target_length=50)
    print(f"After crop (50 frames): {aug_kp.shape}")
    
    aug_kp = augmenter.rotate_2d(dummy_kp, angle_deg=15)
    print(f"After rotation: {aug_kp.shape}")
    
    aug_kp = augmenter(dummy_kp)
    print(f"After random augmentations: {aug_kp.shape}")
    
    print("All tests passed!")
