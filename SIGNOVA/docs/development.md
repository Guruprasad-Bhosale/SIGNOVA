# SIGNOVA Development & Setup Guide

Environment configuration and development commands for Windows 11 and NVIDIA GPU workstations.

---

## 1. System Requirements & Hardware Context

- **OS**: Windows 10/11 (PowerShell 7+ or Windows PowerShell) / Linux / WSL2
- **Python**: 3.11 or 3.12
- **Node.js**: v18+ (Detected: v24.20.0 with npm 11.19.0)
- **Primary GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)

---

## 2. Python Environment Setup

### 2.1 Create Virtual Environment (Optional but Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.2 Core & Development Dependencies
```powershell
pip install -r requirements-dev.txt
```

### 2.3 PyTorch with CUDA 12.1 for RTX 3050
To enable GPU acceleration on the NVIDIA RTX 3050:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(PyTorch will automatically run in CPU mode if CUDA is not installed).*

---

## 3. Key Development Commands

### 3.1 System & Environment Diagnostic
```powershell
python scripts/check_environment.py
```

### 3.2 Dataset Audit & Inspection
```powershell
python scripts/inspect_dataset.py
```

### 3.3 Generate Manifests
```powershell
python scripts/create_manifest.py --dataset isltranslate --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### 3.4 Run Automated Test Suite
```powershell
python -m pytest
```

### 3.5 Launch FastAPI Backend
```powershell
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3.6 Launch React Web Frontend
```powershell
cd apps/web
npm install
npm run dev
```
Web App UI: [http://localhost:5173](http://localhost:5173)

---

## 4. VRAM-Aware GPU Strategy (6 GB Constraints)

1. **Batch Sizing**: Keep local training batch size to $\le 4$ with gradient accumulation steps $\ge 4$ to achieve an effective batch size of 16 without Out-Of-Memory (OOM) errors.
2. **Mixed Precision**: Always set `mixed_precision: true` (`torch.cuda.amp.autocast`) to reduce memory footprint by ~50%.
3. **Feature Pre-Extraction**: Never train directly on raw video frames in memory. Extract 543 MediaPipe keypoints into `.npy` feature files first.
