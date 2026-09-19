# Research Limitations & Supervision Status: SIGNOVA Phase 5

## 1. Distinction Between Continuous Tasks

1. **Continuous Sign Recognition (CSR)**:
   - Maps continuous video to ordered sign/gloss sequences:
     `Video -> ['NAMASTE', 'NAME', 'YOUR', 'WHAT']`
   - Requires: Ordered sign/gloss sequence annotations.
2. **Continuous Sign Translation (SLT)**:
   - Maps continuous video directly to natural language text:
     `Video -> "What is your name?"`
   - Supported by: `ISLTranslate` sentence-level English text pairs.

---

## 2. Phase 5 Specific Limitations

1. **Supervision Blocker**:
   - `ISLTranslate` provides sentence-level English translations, not token-level ISL sign glosses.
   - Real-data CTC training is **BLOCKED** and not claimed.
2. **Adapter Status**:
   - `GenericSequentialAdapter` is verified on synthetic fixtures and labeled `SYNTHETIC_FIXTURE_VALIDATED`.
3. **Transfer Analysis Scope**:
   - Transfer of Phase 3/4 pretrained continuous backbones was analyzed through representation diagnostics (variance, temporal smoothness), not downstream CSR recognition accuracy.
4. **No Translation Claim**:
   - Phase 5 does not claim English translation. English translation belongs to subsequent sequence-to-sequence translation phases.
