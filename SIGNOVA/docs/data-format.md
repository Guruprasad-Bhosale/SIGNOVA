# SIGNOVA Data Format Specification

## 1. Data Lifecycle and Boundaries

SIGNOVA strictly enforces immutable stage boundaries for all data processing:

```
[External Sources] (Hugging Face / Zenodo / Reference Repositories — Read-Only)
       │
       ▼
data/raw/           <- Downloaded raw video/metadata archives (Unmounted in Phase 1)
       │
       ▼
data/interim/       <- Standardized video transcodes (30fps, 720p, uniform codec)
       │
       ▼
data/processed/     <- Cleaned sample sequences and segment boundaries
       │
       ▼
data/features/      <- Extracted MediaPipe holistic landmark arrays (.npy/.npz)
       │
       ▼
data/manifests/     <- Canonical CSV/JSON manifests referencing features & labels
       │
       ▼
[Model Training Pipeline]
```

---

## 2. Canonical Sample Representation (`SignSample`)

Every data point in SIGNOVA adheres to the following contract:

```python
class SampleAvailability(str, Enum):
    REMOTE_ONLY = "REMOTE_ONLY"
    LOCAL_AVAILABLE = "LOCAL_AVAILABLE"
    BOTH = "BOTH"
    MISSING = "MISSING"

@dataclass
class SignSample:
    sample_id: str
    dataset: str
    video_reference: str
    split: str = "train"
    local_video_path: Optional[str] = None
    features_path: Optional[str] = None
    source_language: str = "Indian Sign Language (ISL)"
    target_language: str = "English"
    source_annotation: Optional[str] = None
    target_translation: Optional[str] = None
    gloss_sequence: Optional[List[str]] = None
    signer_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    availability: SampleAvailability = SampleAvailability.REMOTE_ONLY
    metadata: Dict[str, Any]
```

---

## 3. Landmark Feature Topology

- **Pose Landmarks (33 points)**: Key skeletal joints.
- **Face Landmarks (468 points)**: Dense facial mesh for grammatical non-manual markers.
- **Left Hand Landmarks (21 points)**: Hand joints.
- **Right Hand Landmarks (21 points)**: Hand joints.

**Total**: $33 + 468 + 21 + 21 = 543$ landmarks per frame with shape $(T, 543, 3)$.
- Normalization: Center of mass anchored on mid-hip coordinates $(P_{23} + P_{24})/2$, scaled by Euclidean distance $\|P_{11} - P_{12}\|_2$ between shoulders.
