# AI-Based Early Lung Cancer Detection and Classification from Genomic Signals

**Academic Affiliation**: Siddaganga Institute of Technology, Tumakuru, Karnataka  
**Department**: Computer Science & Engineering | Academic Year 2025–26  
**Engineering Team**: Shreeya H M (1SI23CS167), Vidhyashree U (1SI23CS194), Shrinivas Bhagavantagoudar (1SI23CS169), Shrishantha (1SI23CS170)  
**Project Guide**: Mr. Gururaj S P (Assistant Professor, Dept. of CSE)

---

## 📌 Project Overview & Problem Statement

Early detection of lung cancer dramatically enhances patient survival rates from under 15% to over 60%, yet conventional diagnostic modalities (chest radiography and symptomatic presentation) frequently identify malignancies only at advanced, metastatic stages (Stage III/IV).

This project investigates the utilization of **1D genomic signals extracted from DNA sequences** (such as circulating tumor ctDNA liquid biopsies or tissue aspirates) for early-stage lung cancer detection and classification through digital signal processing (DSP) and deep learning.

### Formal Project Objectives:
1. **Objective 1**: Develop an AI-driven computational framework for detecting early-stage lung cancer–related somatic mutations and genomic alterations using **Electron-Ion Interaction Pseudopotential (EIIP)** encoding.
2. **Objective 2**: Design and implement a robust **PyTorch Bidirectional LSTM model with Attention pooling** capable of detecting and categorizing lung cancer into clinically relevant subtypes (Normal Wild-Type, TP53 Somatic Alterations, EGFR Activating Mutations, KRAS Driver Mutations).
3. **Objective 3**: Perform a systematic comparative study between AI models trained on **genomic signals** and those trained on **medical imaging CT scans** ([`dorsar/lung-cancer-detection`](https://huggingface.co/dorsar/lung-cancer-detection)), highlighting their complementary capabilities for precision oncology.

---

## 🔬 End-to-End Diagnostic Pipeline

$$\text{Input DNA Sequence} \xrightarrow{\text{EIIP}} \text{1D Physical Signal} \xrightarrow{\text{FFT + Hilbert}} \text{Spectral \& Envelope Features} \xrightarrow{\text{K-mer Graph}} \text{BiLSTM + Attention} \xrightarrow{} \text{Prediction (+ve / -ve)}$$

### 1. Numerical Encoding: EIIP
Converts nucleotide bases into physical-chemical valence electron energy potentials (in Rydberg units):
* **Adenine (A)**: `0.1260`
* **Cytosine (C)**: `0.1340`
* **Guanine (G)**: `0.0806`
* **Thymine (T)**: `0.1335`

### 2. Feature Extraction: FFT & Hilbert Transform
* **Fast Fourier Transform (FFT)**: Computes the Power Spectral Density (PSD) $|X(f)|^2$. Identifies the characteristic **3-base periodicity** ($f = 1/3 \approx 0.333$), an evolutionary hallmark of protein-coding exons disrupted by somatic frameshift and nonsynonymous mutations.
* **Hilbert Transform**: Constructs the analytic signal $z[n] = x[n] + j \mathcal{H}\{x[n]\}$, extracting the **instantaneous amplitude envelope** $|z[n]|$ and instantaneous frequency $\Delta\phi/2\pi$. Localized spikes reveal mutation clusters and nucleotide instability.

### 3. Topological K-Mer Transition Graph
Constructs a directed graph $G = (V, E)$ where nodes represent $k$-mer motifs ($k=3$) and directed edges represent sequential transitions along the sequence. Extracts graph density, clustering coefficient, spectral radius, and transition Shannon entropy.

### 4. Deep Learning Model: PyTorch BiLSTM with Attention
* Multi-channel feature matrix: $[EIIP(t), \text{Envelope}(t), \text{InstFreq}(t), \text{LocalEnergy}(t)] \in \mathbb{R}^{500 \times 4}$.
* 2-layer Bidirectional LSTM capturing both $5' \rightarrow 3'$ and $3' \rightarrow 5'$ biological dependencies.
* **Self-Attention Pooling**: Generates an attention weight profile across sequence loci, pinpointing the exact nucleotide positions driving the malignant prediction.
* **Dual Output Heads**: Binary classification (+ve Cancer vs. -ve Normal) + Subtype classification (TP53, EGFR, KRAS, Normal).

---

## 📊 Objective 3: Comparative Study (Genomic AI vs. CT Imaging)

Benchmark model: [`dorsar/lung-cancer-detection`](https://huggingface.co/dorsar/lung-cancer-detection) (ResNet50 trained on the IQ-OTH/NCCD chest CT dataset).

| Dimension | Genomic AI Framework (Our Model) | Medical Imaging Model (`dorsar/lung-cancer`) |
| :--- | :--- | :--- |
| **Diagnostic Modality** | Molecular / Circulating Tumor DNA (ctDNA) | Computed Tomography (CT) Scans |
| **Input Data** | 1D Nucleotide Sequences (`A, C, G, T`) | 2D/3D Axial CT Slices (Pixels) |
| **Optimal Stage** | **Stage I (Pre-symptomatic / Microscopic)** | **Stage II - IV (Visible Nodules > 3-5mm)** |
| **Subtyping Precision** | **Targetable Mutations (*EGFR*, *KRAS*, *TP53*)** | Histopathological (*Adeno, Squamous, Large Cell*) |
| **Invasiveness** | Minimally invasive (Liquid biopsy / blood draw) | Non-invasive (Ionizing radiation) |
| **Model Architecture** | EIIP + FFT/Hilbert + PyTorch BiLSTM | Deep Convolutional ResNet50 |
| **Test Accuracy** | ~88.3% - 92.0% | ~98.0% |
| **Inference Latency** | < 45 ms per sequence | ~120 ms per slice |

---

## 📂 Repository Structure

```
lungcancer_dl/
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # Model training & validation pipeline script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── src/
│   ├── encoding.py             # EIIP numerical mapping & validation
│   ├── signal_features.py      # FFT spectral & Hilbert analytic signal processing
│   ├── kmer_graph.py           # K-mer graph generator & topological metrics
│   ├── model.py                # PyTorch GenomicLSTMClassifier & attention pooling
│   ├── dataset.py              # Reference genomic sequences & COSMIC mutation injection
│   └── ct_model.py             # CT scan inference wrapper (dorsar/lung-cancer-detection)
├── models/
│   ├── genomic_lstm.pth        # Trained PyTorch BiLSTM weights
│   ├── model_metadata.json     # Validation metrics, ROC-AUC coordinates & history
│   └── lung_cancer_detection_model.pth # Downloaded CT ResNet50 weights
└── sample_data/
    ├── sample_dataset.csv      # Benchmark curated genomic dataset
    ├── sample_sequences.fasta  # Ready-to-test FASTA sequences
    └── ct_scans/               # Sample CT slices (Normal, Adeno, Large Cell, Squamous)
```

---

## 🚀 Getting Started

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Train the Genomic Model (Optional - Pre-trained weights provided)
```bash
python train.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
