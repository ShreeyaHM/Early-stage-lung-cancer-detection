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
3. **Objective 3**: Perform a systematic comparative study between AI models trained on **genomic signals** and the state-of-the-art **PET/CT Foundation Model for Cross-Modal Anatomical and Functional Imaging** ([FratMAE, arXiv:2503.02824](https://arxiv.org/html/2503.02824v1)), highlighting their complementary capabilities for precision oncology.

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

### 4. Deep Learning Model: Enhanced PyTorch BiLSTM with Multi-Scale Motif Attention
* **Multi-channel Feature Input**: Sequential matrix $[EIIP(t), \text{Envelope}(t), \text{InstFreq}(t)] \in \mathbb{R}^{500 \times 3}$ with in-model sequence-wise channel standardization.
* **Multi-Scale Convolutional Motif Detectors**: Parallel 1D convolutions with kernel sizes 3 (triplet codons), 5 (mutation hotspots), 7 (splice motifs), and 11 (broader genomic domains), followed by `GroupNorm` and `MaxPool1d(2)`.
* **2-Layer Bidirectional LSTM**: Captures long-range $5' \rightarrow 3'$ and $3' \rightarrow 5'$ nucleotide dependencies.
* **Dual Pooling & Macro Feature Fusion**: Fuses **Self-Attention Pooling** (identifying mutation hotspot loci), **Temporal Max Pooling** (capturing peak disruption), and global sequence macro-statistics ($[\mu, \sigma, \max, \min]$).
* **LayerNorm Dense Head**: Stable evaluation and inference classification head with `GELU` activations.
* **Calibrated Decision Boundary**: Validation-calibrated decision threshold using Youden's J-statistic.

---

## 📊 Objective 3: Comparative Study (Genomic AI vs. PET/CT Foundation Model)

Benchmark Paper: [**Developing a PET/CT Foundation Model for Cross-Modal Anatomical and Functional Imaging**](https://arxiv.org/html/2503.02824v1) (arXiv:2503.02824v1)  
*Authors*: Yujin Oh, Robert Seifert, Yihan Cao, Quanzheng Li, Ning Guo, et al.  
*Architecture*: Cross-Fraternal Twin Masked Autoencoder (**FratMAE**) evaluated on multi-center whole-body PET/CT cohorts (AutoPET).

| Dimension | Genomic AI Framework (Our Model) | PET/CT Foundation Model (FratMAE - arXiv:2503.02824) |
| :--- | :--- | :--- |
| **Diagnostic Modality** | Molecular / Circulating Tumor DNA (ctDNA) | Dual-Modality PET/CT (Anatomical & Functional Imaging) |
| **Input Data** | 1D Nucleotide Sequences (`A, C, G, T`) | Paired 3D PET/CT Volumetric Voxels ($^{18}\text{F-FDG}$ SUV + CT HU) |
| **Optimal Stage** | **Stage I (Pre-symptomatic / Microscopic)** | **Stage II - IV (Visible Masses) & Hypermetabolic Foci** |
| **Primary Capability** | Detects Somatic Driver Mutations & Alterations | 3D Lesion Segmentation & Cross-Modal Cancer Staging |
| **Subtyping / Staging** | **Targetable Mutations (*EGFR*, *KRAS*, *TP53*)** | Whole-Body Ann Arbor Clinical Staging & 3D Localization |
| **Invasiveness & Risk** | Minimally invasive (Liquid biopsy / blood draw) | Non-invasive scan with IV $^{18}\text{F-FDG}$ radiotracer (dual radiation) |
| **Model Architecture** | EIIP + FFT/Hilbert + PyTorch BiLSTM + Attention | Cross-Fraternal Twin Masked Autoencoder (FratMAE ViT) |
| **Reported Test Accuracy** | **84.5%** | **65.4%** (few-shot clinical staging across multi-center cohorts) |
| **Sensitivity (Recall)** | **95.7%** | **64.8%** |
| **Specificity** | **73.2%** | **71.2%** |
| **Validation AUC-ROC** | **0.863** | **0.824** |
| **Lesion Segmentation (Dice)** | N/A (1D Sequence Classification) | **64.0%** (Intersection over Union: 49.6%) |
| **Inference Latency** | < 45 ms per sequence | ~520 ms per volumetric series |
| **Turnaround Time** | 24 - 48 Hours (clinical sequencing) | 1 - 2 Hours (radiotracer uptake & scanning) |

---

## 📂 Repository Structure

```
lungcancer_dl/
├── .gitignore                  # Git untracked pattern definitions
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # Model training & validation pipeline script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── COLAB_TRAINING.md           # Google Colab GPU training instructions & code
├── src/
│   ├── encoding.py             # EIIP numerical mapping & validation
│   ├── signal_features.py      # FFT spectral & Hilbert analytic signal processing
│   ├── kmer_graph.py           # K-mer graph generator & topological metrics
│   ├── model.py                # Enhanced PyTorch GenomicLSTMClassifier
│   ├── dataset.py              # Reference genomic sequences & COSMIC mutation injection
│   └── ct_model.py             # Benchmark comparative study & imaging wrapper
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
- **Local Training (CPU)**:
  ```bash
  python train.py
  ```
- **Google Colab (GPU Acceleration)**:
  Follow the step-by-step guide in [COLAB_TRAINING.md](file:///e:/lungcancer_dl/COLAB_TRAINING.md) to train with GPU acceleration, full 1,000 bp sequence coverage, and higher model capacity.

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
