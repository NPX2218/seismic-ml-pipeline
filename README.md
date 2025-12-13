# Seismic ML Data Pipeline

Feature extraction pipeline for earthquake prediction.

## Pipeline Overview

```
Raw Data → Sliding Windows → Bandpass Filter → Feature Extraction → Feature Matrix
                                   │
                                   ├─→ HF1 (5-25 Hz)   rapid vibrations
                                   ├─→ HF2 (0.1-5 Hz)  medium motion
                                   └─→ LF  (residual)  slow drift
```
