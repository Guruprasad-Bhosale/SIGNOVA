"""
Unit tests for dataset adapters.
"""

from pathlib import Path
import pytest
from signova.data.include import INCLUDEAdapter
from signova.data.isltranslate import ISLTranslateAdapter


def test_isltranslate_adapter_interface():
    # Test with non-existent root to verify graceful error handling
    adapter = ISLTranslateAdapter(csv_path=Path("non_existent.csv"))
    val = adapter.validate()
    assert val["valid"] is False


def test_include_adapter_unmounted():
    adapter = INCLUDEAdapter(root_dir=Path("non_existent_include_dir"))
    val = adapter.validate()
    assert val["valid"] is False
    assert val["status"] == "unmounted"
