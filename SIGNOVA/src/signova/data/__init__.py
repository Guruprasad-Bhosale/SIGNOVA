"""
Dataset ingestion, manifest management, and adapters for SIGNOVA.
"""

from signova.data.base import DatasetAdapter
from signova.data.cache import VideoCacheManager
from signova.data.collate import PadCollate
from signova.data.feature_dataset import SignLanguageFeatureDataset
from signova.data.include import INCLUDEAdapter
from signova.data.isltranslate import ISLTranslateAdapter
from signova.data.manifest import Manifest, ManifestEntry
from signova.data.models import SampleAvailability, SignSample
from signova.data.remote import RemoteVideoResolver
from signova.data.schema import SignSampleSchema
from signova.data.splits import deterministic_split
from signova.data.validation import DataValidationError, validate_manifest, validate_manifest_entry

from signova.data.continuous_dataset import ContinuousPadCollate, ContinuousSignDataset
from signova.data.windowing import SlidingWindowExtractor, verify_window_split_leakage

__all__ = [
    "DatasetAdapter",
    "INCLUDEAdapter",
    "ISLTranslateAdapter",
    "Manifest",
    "ManifestEntry",
    "SampleAvailability",
    "SignSample",
    "SignSampleSchema",
    "deterministic_split",
    "DataValidationError",
    "validate_manifest",
    "validate_manifest_entry",
    "SignLanguageFeatureDataset",
    "PadCollate",
    "VideoCacheManager",
    "RemoteVideoResolver",
    "ContinuousSignDataset",
    "ContinuousPadCollate",
    "SlidingWindowExtractor",
    "verify_window_split_leakage",
]

