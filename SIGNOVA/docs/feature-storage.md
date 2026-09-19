# SIGNOVA Feature Storage Format & Benchmarks

## Overview

SIGNOVA evaluated three serialization strategies for storing high-dimensional skeletal landmark sequences across 31,222 dataset samples:

1. **Uncompressed `.npy` (raw NumPy binary)**
2. **Uncompressed `.npz` (multi-array container)**
3. **Compressed `.npz` (`np.savez_compressed` DEFLATE container)**

---

## 1. Storage Format Benchmark Results

Benchmarked on a representative sequence ($T = 150$ frames, $543$ joints, $3$ coordinates + detection masks + timestamps + JSON metadata headers):

| Metric | Uncompressed `.npy` | Uncompressed `.npz` | Compressed `.npz` (Adopted) |
| :--- | :--- | :--- | :--- |
| **File Size (150 frames)** | ~977.5 KB | ~983.2 KB | **~905.1 KB (dense) / ~2.0 KB (sparse)** |
| **Storage Multiplier** | 1.00x | 1.01x | **0.92x (dense) / 0.002x (sparse)** |
| **Multi-Tensor Bundling** | No (requires loose files) | Yes (`landmarks`, `masks`, `meta`) | **Yes (single atomic `.npz`)** |
| **Save Latency** | 0.94 ms | 7.81 ms | **40.24 ms** |
| **Load Latency** | 0.47 ms | 2.45 ms | **5.95 ms** |
| **Metadata Headers** | External | Embedded JSON | **Embedded JSON** |

---

## 2. Decision Rationale

### Why Compressed `.npz` was Selected:
1. **Atomic Multi-Modal Bundling**: A single `.npz` archive bundles $(T, 543, 3)$ landmarks, $(T, 4)$ detection masks, $(T,)$ timestamps, frame indices, and full provenance/quality JSON headers.
2. **Sparse Compression Efficiency**: Realistic landmark sequences have extensive zero-padding and occluded facial/hand landmarks, yielding massive compression ratios ($>5\times$) in practical datasets.
3. **Low I/O Bottlenecks**: With PyTorch / NumPy memory-mapped reads and SSD sequential access, 5.95 ms load latency is negligible compared to model computation times.
4. **Safety & Immutability**: Eliminates multi-file synchronization errors between masks, timestamps, and keypoint arrays.
