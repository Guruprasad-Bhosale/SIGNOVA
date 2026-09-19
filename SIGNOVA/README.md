# 🤟 SIGNOVA

### A Real-Time Computer Vision Pipeline for Indian Sign Language

<p align="center">
  <strong>See the Sign. Understand the Sequence. Build the Evidence.</strong><br/>
  A research-oriented system for real-time Indian Sign Language (ISL) processing using computer vision, hand/pose landmarks, temporal modeling, and rigorously verified evaluation.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20%2F%20Development-8b5cf6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision-6366f1?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Focus-Indian%20Sign%20Language-111827?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/MediaPipe-Landmarks-FF6F00?style=for-the-badge"/>
</p>

---

# 🧠 What is SIGNOVA?

**SIGNOVA** is an experimental, research-oriented computer vision system focused on **Indian Sign Language (ISL)**.

The project explores the complete pipeline required to move from:

```text
Camera Input
     │
     ▼
Visual Processing
     │
     ▼
Human / Hand Landmark Extraction
     │
     ▼
Temporal Feature Sequences
     │
     ▼
Sequence Modeling
     │
     ▼
ISL Recognition
     │
     ▼
Evaluation
     │
     ▼
Potential Translation
```

The core philosophy is simple:

> **A sign is not merely a frame. It is a temporal sequence of human movement.**

SIGNOVA therefore focuses on preserving temporal information and building a scientifically defensible path toward continuous sign-language recognition.

---

# 🎯 The Problem

Sign-language recognition is significantly more difficult than classifying isolated images.

A real signing sequence contains:

- 🤚 Hand configuration
- 📍 Hand position
- ↔️ Movement
- 🧍 Body pose
- 👤 Interaction between both hands
- ⏱️ Temporal dynamics
- 🔗 Relationships between consecutive signs

A single image can describe **what a hand looks like**.

It cannot fully describe:

> **how the sign was performed through time.**

SIGNOVA therefore treats signing as a **sequence modeling problem**.

---

# 🔬 Core Research Direction

The project is built around a temporal representation:

```text
Video
  │
  ▼
Frames
  │
  ▼
MediaPipe Landmark Extraction
  │
  ├── Left Hand
  ├── Right Hand
  └── Pose
  │
  ▼
Feature Vector
  │
  ▼
Temporal Sequence
  │
  ▼
CTC-Based Sequence Model
  │
  ▼
ISL Recognition
```

This architecture allows the system to move beyond static gesture classification toward continuous sequence recognition.

---

# 👁️ Real-Time Vision Pipeline

SIGNOVA currently contains an operational live vision pipeline built around:

```text
Camera
   │
   ▼
OpenCV
   │
   ▼
MediaPipe
   │
   ├── Left Hand
   ├── Right Hand
   └── Pose
   │
   ▼
HANDS_POSE Feature Representation
   │
   ▼
64-Frame Temporal Buffer
   │
   ▼
Stride-Based Streaming
   │
   ▼
Live Runtime
```

The runtime is designed to keep visual acquisition and downstream processing modular.

---

# 📐 Landmark Representation

SIGNOVA uses **MediaPipe** to extract human landmarks.

The current `HANDS_POSE` representation captures:

```text
Left Hand
Right Hand
Pose
```

The resulting landmark stream contains **543 landmark points** across the configured hand/pose representation.

Conceptually:

```text
                HUMAN
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
   LEFT HAND   RIGHT HAND    POSE
       │          │          │
       └──────────┼──────────┘
                  ▼
           Feature Vector
                  │
                  ▼
          Temporal Sequence
```

---

# ⏱️ Temporal Processing

SIGNOVA does not treat incoming frames as independent samples.

A temporal buffer is maintained to preserve motion information.

Current live configuration includes:

```text
Buffer Length: 64 frames
Stride:        16 frames
```

This provides a sliding temporal representation:

```text
Time ──────────────────────────────────────>

[ 1 ───────────────────────────── 64 ]
        [ 17 ───────────────────── 80 ]
                [ 33 ─────────────────── 96 ]
```

The approach allows the system to continuously process an incoming stream while retaining temporal context.

---

# ⚡ Live Runtime

The live pipeline currently integrates:

```text
Camera
   │
   ▼
OpenCV
   │
   ▼
MediaPipe
   │
   ▼
SIGNOVA Live Runtime
   │
   ├── Landmark extraction
   ├── Feature construction
   ├── Temporal buffering
   └── WebSocket streaming
   │
   ▼
Live UI
```

The current CPU-only runtime has demonstrated approximately:

```text
~20.8 FPS
```

under the validated development configuration.

This is an engineering measurement of the live pipeline, **not a recognition-accuracy metric**.

---

# 🌐 Real-Time Communication

SIGNOVA uses a WebSocket-based live communication layer to connect the vision runtime with the user interface.

```text
┌──────────────┐
│    Camera    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ OpenCV / MP  │
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│ SignovaLiveRuntime │
└──────────┬─────────┘
           │
        WebSocket
           │
           ▼
┌────────────────────┐
│      Live UI       │
└────────────────────┘
```

This keeps the capture/processing layer separate from presentation.

---

# 🧠 Why CTC?

Continuous Sign Language Recognition requires a model that can map a sequence of visual features to a sequence of linguistic units without requiring an exact frame-level alignment for every output token.

SIGNOVA therefore investigates **Connectionist Temporal Classification (CTC)**.

Conceptually:

```text
Input Sequence

x₁  x₂  x₃  x₄  x₅  x₆  x₇  x₈
│   │   │   │   │   │   │   │
▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼

Temporal Encoder
        │
        ▼

CTC Output
│  │  │  │  │  │  │  │
▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼

blank  ISL  blank  SIGN  blank ...

        │
        ▼

CTC Decoding
        │
        ▼

Recognized Sequence
```

The intended model direction includes a **BiGRU-CTC** architecture for genuine sequential recognition once the required verified training data is available.

---

# 🧪 Dataset Integrity Comes First

SIGNOVA deliberately separates:

### Data availability

from

### Data validity

and from

### Model validity.

A dataset cannot be treated as training-ready simply because video files exist.

For genuine sequence recognition, the project requires:

```text
Video
  │
  ├── Verified identity
  ├── Verified gloss / annotation
  ├── Sequential alignment information
  ├── Provenance
  └── Leakage checks
       │
       ▼
Canonical Dataset Gate
       │
       ▼
CTC Training
```

This prevents unsupported recognition claims from entering the system.

---

# 🛡️ Scientific Guardrails

SIGNOVA follows strict evaluation safeguards.

### No Random-Split Shortcut

Temporal video data can easily leak information between training and evaluation.

Therefore, dataset partitioning must be performed deliberately and verified.

### No Unverified Glosses

Videos without trustworthy sequential annotations are not treated as valid ground truth for genuine CTC training.

### No Synthetic Recognition Claims

A functional camera pipeline is **not** presented as a trained ISL recognition model.

### No Metric Without Valid Evaluation

Accuracy, F1, TER, S-I-D, or similar metrics are reported only when generated from a valid held-out evaluation protocol.

---

# 📊 Evaluation Philosophy

SIGNOVA separates system evaluation into different layers.

```text
┌──────────────────────────────┐
│       SYSTEM EVALUATION      │
├──────────────────────────────┤
│                              │
│  Vision Pipeline             │
│       ↓                      │
│  Landmark Extraction         │
│       ↓                      │
│  Temporal Processing         │
│       ↓                      │
│  Model Training              │
│       ↓                      │
│  Held-Out Recognition        │
│       ↓                      │
│  Translation                 │
│                              │
└──────────────────────────────┘
```

This prevents infrastructure performance from being confused with recognition quality.

For genuine sequential evaluation, the project targets metrics appropriate to the recognition task, including:

```text
TER
S-I-D
F1
```

alongside appropriate dataset and split documentation.

---

# 🔐 Provenance & Reproducibility

A major part of SIGNOVA is ensuring that experimental results can be traced back to their source.

The system uses provenance-oriented artifacts and immutable gate snapshots to preserve the state of an evaluated dataset and experiment.

Conceptually:

```text
Dataset
   │
   ▼
Verification
   │
   ▼
Canonical Gate
   │
   ▼
Immutable Snapshot
   │
   ▼
Training Run
   │
   ▼
Evaluation
   │
   ▼
Reported Result
```

The objective is to make it possible to answer:

> **"Where did this model and this metric come from?"**

---

# 🧱 Architecture

The project is organized into independently verifiable layers.

```text
SIGNOVA/
│
├── data/
│   ├── acquisition/
│   ├── verification/
│   ├── annotations/
│   └── provenance/
│
├── vision/
│   ├── camera/
│   ├── mediapipe/
│   ├── landmarks/
│   └── features/
│
├── temporal/
│   ├── buffering/
│   └── sequencing/
│
├── models/
│   ├── ctc/
│   ├── encoder/
│   └── decoding/
│
├── evaluation/
│   ├── splits/
│   ├── metrics/
│   └── reports/
│
├── runtime/
│   ├── live/
│   └── websocket/
│
├── ui/
│
├── tests/
│
└── docs/
```

> Directory names may evolve as implementation continues.

---

# 🔄 End-to-End Architecture

```text
                    ┌───────────────┐
                    │    CAMERA     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    OpenCV     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   MediaPipe   │
                    └───────┬───────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │  HANDS_POSE       │
                  │  Representation   │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Temporal Buffer   │
                  │ 64 frames / 16    │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Sequence Encoder  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │     CTC Head      │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │     Decoder       │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Recognition Layer │
                  └───────────────────┘
```

---

# 🧪 Testing

SIGNOVA maintains automated verification across the system.

The current validated development state includes:

```text
417 tests passing
44 / 44 reference checks intact
```

Testing covers infrastructure, data contracts, runtime behavior, and project invariants.

The testing philosophy is:

```text
Code Change
    │
    ▼
Automated Tests
    │
    ▼
Invariant Checks
    │
    ▼
Phase Gate
    │
    ▼
Next Stage
```

---

# 🚦 Current System State

SIGNOVA explicitly distinguishes between system states.

### STATE_B — Operational Vision / Research Pipeline

The current validated state includes:

```text
✓ Camera pipeline
✓ OpenCV processing
✓ MediaPipe landmarks
✓ HANDS_POSE representation
✓ Temporal buffering
✓ WebSocket live communication
✓ Live UI integration
✓ Runtime verification
✓ Automated test suite
✓ Provenance infrastructure
```

However:

```text
✗ Genuine CTC recognition claim
✗ Validated continuous recognition metrics
✗ Production translation claim
```

These remain gated behind verified sequential training data and leakage-safe evaluation.

---

# 🔬 Research Gate

The transition toward genuine recognition follows:

```text
Verified Sequential Data
          │
          ▼
Canonical Dataset Gate
          │
          ▼
Immutable Snapshot
          │
          ▼
Leakage-Safe Split
          │
          ▼
BiGRU-CTC Training
          │
          ▼
Held-Out Evaluation
          │
          ▼
Proven Metrics
          │
          ▼
Live-Model Registration
          │
          ▼
Validated Recognition
```

A model being **trained** does not automatically make it **authorized for live use**.

That distinction is intentional.

---

# 📈 Development Roadmap

```text
[✓] Phase 0  — Project Foundation
[✓] Phase 1  — Architecture & Core Contracts
[✓] Phase 2  — Data & Dataset Infrastructure
[✓] Phase 3  — Verification & Provenance
[✓] Phase 4  — Feature Representation
[✓] Phase 5  — Temporal Processing
[✓] Phase 6  — Runtime Infrastructure
[✓] Phase 7  — Camera Integration
[✓] Phase 8  — MediaPipe Integration
[✓] Phase 9  — Sequential Dataset Acquisition & Verification
[✓] Phase 10+ — Research Pipeline Hardening
[✓] Phase 20 — Live Runtime & WebSocket Integration
[→] Phase 21 — Genuine Sequential ISL CTC Training & Evaluation
[→] Future   — Validated Live Recognition
[→] Future   — Translation Research
```

The project progresses through explicit gates rather than treating every implemented subsystem as a validated research result.

---

# 🛠️ Technology Stack

### Computer Vision

```text
Python
OpenCV
MediaPipe
```

### Machine Learning

```text
PyTorch
BiGRU
CTC
Sequence Modeling
```

### Real-Time Runtime

```text
WebSocket
Live Camera Processing
Temporal Buffers
```

### Engineering

```text
Automated Testing
Provenance Tracking
Dataset Verification
Experiment Snapshots
```

---

# 🚀 Running SIGNOVA

## Prerequisites

```text
Python 3.x
Git
Camera / Webcam
Required Python dependencies
```

## Clone

```bash
git clone https://github.com/YOUR_USERNAME/signova.git
cd signova
```

## Create Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python <entrypoint>
```

> Replace `<entrypoint>` with the current project runtime entry point.

---

# 📚 Research Principles

SIGNOVA is built around several principles.

### 01 — Don't confuse a demo with a model.

A working camera and landmark pipeline demonstrates infrastructure—not recognition accuracy.

### 02 — Don't train on unverified ground truth.

Incorrect annotations produce misleading models.

### 03 — Don't evaluate with leakage.

A high score is meaningless if information from the evaluation set leaks into training.

### 04 — Don't hide uncertainty.

When evidence is insufficient, the system should remain in a limited state rather than manufacture confidence.

### 05 — Every result should have provenance.

A metric without a reproducible origin is difficult to trust.

---

# 🌐 The Vision

The long-term vision for SIGNOVA is to create a complete pipeline capable of moving from live visual signing toward validated continuous ISL recognition.

```text
             HUMAN SIGNER
                  │
                  ▼
              📷 CAMERA
                  │
                  ▼
           👁️ VISUAL SYSTEM
                  │
                  ▼
          🖐️ LANDMARK STREAM
                  │
                  ▼
          ⏱️ TEMPORAL FEATURES
                  │
                  ▼
          🧠 SEQUENCE MODEL
                  │
                  ▼
           🤟 ISL RECOGNITION
                  │
                  ▼
          📝 LINGUISTIC OUTPUT
                  │
                  ▼
          🌐 FUTURE TRANSLATION
```

The objective is not to claim that every stage already works.

The objective is to build the scientific and engineering infrastructure required to make each stage **measurable, reproducible, and defensible**.

---

# ⭐ Why SIGNOVA?

Sign-language technology sits at the intersection of:

```text
Computer Vision
       +
Human Pose Estimation
       +
Temporal Modeling
       +
Machine Learning
       +
Accessibility
       +
Real-Time Systems
```

SIGNOVA brings these areas together into a single research pipeline while maintaining explicit boundaries between:

**what is implemented, what is experimentally validated, and what remains future work.**

---

# 📌 Project Status

> 🟣 **SIGNOVA is an active research and engineering project.**

The real-time visual processing pipeline is operational.

The project is currently advancing toward genuine sequential ISL recognition through verified data, CTC training, held-out evaluation, provenance tracking, and controlled live-model integration.

Recognition and translation claims are intentionally withheld until the required evidence exists.

---

<p align="center">

# 🤟 SIGNOVA

### **See the movement. Model the sequence. Verify the result.**

<br/>

**Indian Sign Language × Computer Vision × Sequence Learning**

<br/>

⭐ Star the repository to follow the research.

</p>
