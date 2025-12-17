<h1 align="center">
Seismic ML Data Pipeline
</h1>

<p align="center">
A signal processing pipeline for extracting machine learning features from seismic waveform data
</p>

## Description
This pipeline takes seismic time series data recorded from the Alaska Earthquake Center and outputs features to be used in a deep learning model for earthquake prediction. Rather than using a spectrogram, this pipeline converts waveform data into histograms of amplitude distributions within each frequency band. 

## Pipeline Architecture
```
Raw Data → Sliding Windows → Bandpass Filter → Feature Extraction → Feature Matrix
                                   │
                                   ├─→ HF1 (5-25 Hz)   rapid vibrations
                                   ├─→ HF2 (0.1-5 Hz)  medium motion
                                   └─→ LF  (residual)  slow drift
```

## Feature Extraction Procedure
...

## Installation
```bash
git clone https://github.com/NPX2218/seismic-ml-pipeline.git
cd seismic-ml-pipeline
pip install -r requirements.txt
```
## Execution
```bash
python3 main.py
```


## Acknowledgments
- [NSF SAGE Data](https://www.iris.edu/hq/)
- [NumPy](https://numpy.org/)
- [ObsPy](https://obspy.org/)
- ...
