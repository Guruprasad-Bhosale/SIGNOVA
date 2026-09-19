"""
Unit tests for Phase 10 Extended Provenance Chain with Parent-Child Linkage.
"""

import pytest
from signova.data.provenance import (
    ProvenanceChain,
    ProvenanceNode,
)


def test_phase10_extended_provenance_chain():
    chain = ProvenanceChain(sample_id="phase10_sample_001")

    chain.add_step("SOURCE", "https://data.example.org/dataset")
    chain.add_step("ACQUISITION", "acq_001", parent_provenance_id="https://data.example.org/dataset")
    chain.add_step("DOWNLOAD", "raw_isl_video.mp4", sha256="SHA_DOWNLOAD", parent_provenance_id="acq_001")
    chain.add_step("RAW_ARTIFACT", "data/raw/phase10/sample.mp4", sha256="SHA_RAW", parent_provenance_id="raw_isl_video.mp4")
    chain.add_step("INGESTION", "ingest_001", parent_provenance_id="SHA_RAW")
    chain.add_step("ANNOTATION", "annot_001", sha256="SHA_ANNOT", parent_provenance_id="ingest_001")
    chain.add_step("VIDEO", "derived_video.mp4", parent_provenance_id="annot_001")
    chain.add_step("FEATURE", "derived_features.npz", sha256="SHA_FEAT", parent_provenance_id="derived_video.mp4")
    chain.add_step("MODEL_SAMPLE", "sample_001", parent_provenance_id="derived_features.npz")

    assert chain.is_complete()
    chain_dict = chain.to_dict()
    assert chain_dict["chain_length"] == 9

    # Verify parent-child linkage
    raw_node = [n for n in chain_dict["nodes"] if n["node_type"] == "RAW_ARTIFACT"][0]
    assert raw_node["parent_provenance_id"] == "raw_isl_video.mp4"
