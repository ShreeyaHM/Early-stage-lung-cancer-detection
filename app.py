"""
AI-Based Early Lung Cancer Detection and Classification from Genomic Signals
Interactive Streamlit Web Application
Siddaganga Institute of Technology - Department of Computer Science & Engineering
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from PIL import Image

# Core internal modules
from src.encoding import eiip_encode, clean_dna_sequence, validate_dna_sequence, compute_base_composition
from src.signal_features import compute_fft_features, compute_hilbert_features, extract_combined_signal_matrix
from src.kmer_graph import build_kmer_graph, compute_kmer_graph_metrics, create_kmer_graph_plot
from src.model import GenomicLSTMClassifier, predict_single_sequence
from src.ct_model import predict_ct_scan, get_comparative_matrix, CLASS_NAMES as CT_CLASS_NAMES
from src.dataset import REFERENCE_SEQUENCES, inject_somatic_mutation

# Page Configuration
st.set_page_config(
    page_title="AI Lung Cancer Detection Using Genomic Signals",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 2px;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 22px;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        color: #1e293b;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.09);
    }
    .card h3, .card h4, .card p, .card b {
        color: #1e293b;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2563EB;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-positive {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-negative {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #34D399;
    }
    .dna-seq {
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 2.5px;
        word-wrap: break-word;
        font-size: 1.05rem;
        line-height: 1.7;
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
        color: #0f172a;
        max-height: 220px;
        overflow-y: auto;
    }
    .dna-a { color: #059669; font-weight: bold; } /* Green */
    .dna-c { color: #2563EB; font-weight: bold; } /* Blue */
    .dna-g { color: #D97706; font-weight: bold; } /* Orange */
    .dna-t { color: #DC2626; font-weight: bold; } /* Red */
    .pipeline-step {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.92rem;
        color: #1e293b;
        border-left: 4px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for colored DNA sequence display
def render_dna_sequence(sequence: str) -> str:
    colored_seq = ""
    for char in sequence.upper():
        if char == 'A':
            colored_seq += '<span class="dna-a">A</span>'
        elif char == 'C':
            colored_seq += '<span class="dna-c">C</span>'
        elif char == 'G':
            colored_seq += '<span class="dna-g">G</span>'
        elif char == 'T':
            colored_seq += '<span class="dna-t">T</span>'
        else:
            colored_seq += char
    return f'<div class="dna-seq">{colored_seq}</div>'

# Cache PyTorch genomic model
@st.cache_resource
def load_genomic_model():
    """Load the trained PyTorch model with caching to ensure high performance."""
    model_path = os.path.join("models", "genomic_lstm.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GenomicLSTMClassifier(input_dim=3, hidden_dim=128, num_layers=2, dropout=0.20)
    if os.path.exists(model_path):
        try:
            weights = torch.load(model_path, map_location=device)
            model.load_state_dict(weights)
            model.to(device)
            model.eval()
            return model, True, "PyTorch BiLSTM (EIIP+FFT+Hilbert) Loaded Successfully"
        except Exception as e:
            return model, False, f"Model weights error: {e}"
    return model, False, "Model weights not found. Please run train.py first."

# Cache Model Metadata
@st.cache_data
def load_model_metadata():
    meta_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

genomic_model, has_genomic_model, model_status = load_genomic_model()
metadata = load_model_metadata()

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🧬 Genomic AI Diagnostic")
    st.markdown("---")
    
    pages = [
        "Dashboard",
        "DNA Prediction",
        "Upload DNA File",
        "Analytics",
        "Comparative Study",
        "Tech Stack",
        "Team Details"
    ]
    selected_page = st.radio("Navigation Menu:", pages)
    
    st.markdown("---")
    st.caption("AI-Based Early Lung Cancer Detection")
    st.markdown(f"**Model Status**: `{'Online ✅' if has_genomic_model else 'Offline ⚠️'}`")

# Header
st.markdown('<p class="main-header">🧬 AI-Based Early Lung Cancer Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Classification from Genomic Signals</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------------------
if selected_page == "Dashboard":
    st.markdown("### Project Vision & Objectives")
    st.write(
        "Early detection of lung cancer significantly improves 5-year patient survival rates from under 15% to over 60%. "
        "Conventional diagnostic approaches (like chest radiography and symptoms) often identify the disease only at advanced stages (Stage III/IV). "
        "This project investigates the use of **1D genomic signals extracted from DNA sequences** (such as circulating tumor ctDNA or tissue biopsy) "
        "for early-stage lung cancer detection and classification through advanced computational signal processing and deep learning."
    )
    
    # 3 Stat Cards
    c1, c2 = st.columns(2)
    with c1:
        acc_text = f"{metadata['test_metrics']['accuracy']}%" if metadata else "N/A"
        st.markdown(f'<div class="card"><div class="stat-label">Model Test Accuracy</div><div class="stat-value">{acc_text}</div><p style="margin:0; font-size:0.85rem; color:#64748B;">Validated on unseen sequences</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="stat-label">Imaging Benchmark</div><div class="stat-value">PET/CT Foundation Model</div><p style="margin:0; font-size:0.85rem; color:#64748B;">FratMAE (arXiv:2503.02824)</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Formal Project Objectives")
    
    obj_c1, obj_c2, obj_c3 = st.columns(3)
    with obj_c1:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #2563EB; min-height: 220px;">
            <h4>🎯 Objective 1: Genomic Alterations</h4>
            <p style="font-size: 0.95rem; color: #475569;">
                Develop an AI-driven computational framework for detecting early-stage lung cancer–related somatic mutations 
                and genomic alterations using <b>Electron-Ion Interaction Pseudopotential (EIIP)</b> encoding.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with obj_c2:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #10B981; min-height: 220px;">
            <h4>🎯 Objective 2: DL Classification</h4>
            <p style="font-size: 0.95rem; color: #475569;">
            Design and implement a robust <strong>PyTorch Bidirectional LSTM with Attention pooling</strong>
            to classify genomic signals into <strong>Cancer and Normal classes</strong>, using features
            derived from <strong>EIIP encoding, FFT, and Hilbert Transform</strong>.
        </div>
        """, unsafe_allow_html=True)
        
    with obj_c3:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #F59E0B; min-height: 220px;">
            <h4>🎯 Objective 3: Comparative Study</h4>
            <p style="font-size: 0.95rem; color: #475569;">
                Perform a systematic comparative study between AI models trained on <b>genomic signals</b> and cross-modal foundation models trained on <b>PET/CT anatomical and functional imaging (FratMAE, arXiv:2503.02824)</b>, highlighting complementary capabilities for precision oncology.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### End-to-End Methodological Workflow")
    
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("""
        <div class="pipeline-step"><b>Step 1: Input DNA Sequence</b><br>FASTA/text format containing nucleotide base pairs (A, C, G, T).</div>
        <div class="pipeline-step"><b>Step 2: EIIP Numerical Encoding</b><br>Physical-chemical conversion: A=0.1260, C=0.1340, G=0.0806, T=0.1335 Rydberg units.</div>
        <div class="pipeline-step"><b>Step 3: Fast Fourier Transform (FFT)</b><br>Extracts Power Spectral Density (PSD), 3-base periodicity ratio (coding exon marker), and spectral energy.</div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="pipeline-step"><b>Step 4: Hilbert Transform</b><br>Extracts the analytic signal, instantaneous amplitude envelope, and instantaneous phase/frequency fluctuations.</div>
        <div class="pipeline-step"><b>Step 5: K-Mer Transition Graph</b><br>Constructs topological network graph representing motif transitions, graph density, and Shannon entropy.</div>
        <div class="pipeline-step"><b>Step 6: Deep Learning Inference (BiLSTM)</b><br>2-layer BiLSTM with attention pooling predicts Cancer Positive (+ve) / Normal (-ve).</div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 2: DNA PREDICTION (REAL-TIME PIPELINE)
# -------------------------------------------------------------
elif selected_page == "DNA Prediction":
    st.markdown("### Real-Time DNA Sequence Inference Pipeline")
    st.write(
        "Enter a DNA sequence to execute the complete signal processing and deep learning pipeline."
    )
    
    dna_input = st.text_area(
        "Genomic Nucleotide Sequence (A, C, G, T)",
        value="",
        height=140,
        placeholder="Paste DNA sequence here (e.g., ATGGAGGAGCCGCAGTCAGATCCTAGC...)"
    )
        
    execute_btn = st.button("Execute Pipeline 🚀", type="primary")
    
    if execute_btn:
        is_valid, err_msg = validate_dna_sequence(dna_input)
        if not is_valid:
            st.error(f"❌ Input Validation Error: {err_msg}")
        elif not has_genomic_model:
            st.error(f"❌ Deep Learning Model Unavailable: {model_status}. Please ensure training has completed.")
        else:
            clean_seq = clean_dna_sequence(dna_input)
            
            with st.spinner("Processing EIIP encoding, FFT, Hilbert transform, K-mer graph, and LSTM inference..."):
                # 1. EIIP Encoding
                eiip_sig = eiip_encode(clean_seq, target_length=500)
                
                # 2. FFT Spectral Analysis
                fft_features = compute_fft_features(eiip_sig)
                
                # 3. Hilbert Transform Analysis
                hilbert_features = compute_hilbert_features(eiip_sig)
                
                # 4. K-Mer Graph
                kmer_graph = build_kmer_graph(clean_seq, k=3)
                kmer_metrics = compute_kmer_graph_metrics(kmer_graph)
                
                # 5. Combined Matrix & DL Model Inference
                feat_matrix = extract_combined_signal_matrix(eiip_sig, target_length=500)
                thresh = metadata['test_metrics'].get('optimal_threshold', 0.50) if metadata else 0.50
                pred_result = predict_single_sequence(genomic_model, feat_matrix, threshold=thresh)
                
            # Temporary Debug Logging (Requirement 14)
            print(f"[DEBUG INFERENCE] First 30 bp: {clean_seq[:30]} | Length: {len(clean_seq)} | Feature matrix: {feat_matrix.shape} | Normal prob: {pred_result['normal_probability']}% | Cancer prob: {pred_result['cancer_probability']}%", flush=True)
            with st.expander("🛠️ Inference Debug Diagnostics (Active Run)", expanded=False):
                d_c1, d_c2, d_c3 = st.columns(3)
                d_c1.write(f"**Cleaned Seq (First 30 bp)**: `{clean_seq[:30]}`")
                d_c1.write(f"**Sequence Length**: `{len(clean_seq)} bp`")
                d_c2.write(f"**Feature Matrix Shape**: `{feat_matrix.shape}`")
                d_c2.write(f"**Tensor Input Shape**: `(1, {feat_matrix.shape[0]}, {feat_matrix.shape[1]})`")
                d_c3.write(f"**Normal Probability**: `{pred_result['normal_probability']}%`")
                d_c3.write(f"**Cancer Probability**: `{pred_result['cancer_probability']}%`")
                
            # Display Prediction Banner Cards
            is_cancer = pred_result['is_cancer']
            pred_label = "Cancer" if is_cancer else "Normal"
            pred_color = "#EF4444" if is_cancer else "#10B981"
            risk_level = "High Risk" if is_cancer and pred_result['mutation_burden'] > 30 else ("Moderate Risk" if is_cancer else "Low / Baseline")
            risk_color = "#EF4444" if "High" in risk_level else ("#F59E0B" if "Moderate" in risk_level else "#10B981")
            
            st.markdown("---")
            st.markdown("### 🔬 Molecular Analysis & Clinical Inference")
            
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.markdown(f'<div class="card" style="border-top: 5px solid {pred_color};"><div class="stat-label">Model Diagnosis</div><div class="stat-value" style="color: {pred_color}; font-size: 1.6rem;">{pred_label}</div></div>', unsafe_allow_html=True)
            with res_c2:
                st.markdown(f'<div class="card" style="border-top: 5px solid #2563EB;"><div class="stat-label">Prediction Confidence</div><div class="stat-value" style="font-size: 1.6rem;">{pred_result["confidence_score"]}%</div></div>', unsafe_allow_html=True)
            with res_c3:
                st.markdown(f'<div class="card" style="border-top: 5px solid {risk_color};"><div class="stat-label">Mutational Burden (TMB)</div><div class="stat-value" style="color: {risk_color}; font-size: 1.6rem;">{pred_result["mutation_burden"]}%</div></div>', unsafe_allow_html=True)
                
            # Base composition & viewer
            base_comp = compute_base_composition(clean_seq)
            st.markdown("#### DNA Sequence Base Viewer")
            st.markdown(render_dna_sequence(clean_seq[:800] + ("..." if len(clean_seq) > 800 else "")), unsafe_allow_html=True)
            st.caption(f"Sequence Length: **{base_comp['total_length']} bps** | GC Content: **{base_comp['gc_content']}%** | A: {base_comp['percentages']['A']}% | C: {base_comp['percentages']['C']}% | G: {base_comp['percentages']['G']}% | T: {base_comp['percentages']['T']}%")
            
            # Probability Distribution & Mutation Gauge
            sub_col1, sub_col2 = st.columns([1.2, 1])
            with sub_col1:
                st.markdown("#### Binary Class Probability Distribution")
                df_prob = pd.DataFrame([
                    {'Class': 'Cancer', 'Probability (%)': pred_result['cancer_probability']},
                    {'Class': 'Normal', 'Probability (%)': pred_result['normal_probability']}
                ])
                fig_prob = px.bar(
                    df_prob, x='Probability (%)', y='Class', orientation='h',
                    color='Class',
                    color_discrete_map={'Cancer': '#EF4444', 'Normal': '#10B981'},
                    text='Probability (%)'
                )
                fig_prob.update_layout(height=260, showlegend=False, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_prob, use_container_width=True)
                
            with sub_col2:
                st.markdown("#### Mutational Burden Stratification")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_result['mutation_burden'],
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 10], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [10, 30], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [30, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Signal Visualizations Tabs (EIIP, FFT, Hilbert, K-mer Graph, Attention Map)
            st.markdown("---")
            st.markdown("### 📊 Signal Processing & Deep Feature Representations")
            sig_tab1, sig_tab2, sig_tab3, sig_tab4, sig_tab5 = st.tabs([
                "1. EIIP Signal",
                "2. FFT Power Spectrum",
                "3. Hilbert Envelope",
                "4. K-Mer Transition Graph",
                "5. LSTM Attention Hotspots"
            ])
            
            with sig_tab1:
                st.markdown("##### Electron-Ion Interaction Pseudopotential (EIIP) 1D Genomic Signal")
                df_eiip = pd.DataFrame({'Position (bps)': np.arange(len(eiip_sig[:250])), 'EIIP Energy': eiip_sig[:250]})
                fig_eiip = px.line(df_eiip, x='Position (bps)', y='EIIP Energy', line_shape='hv', title="EIIP Discrete Energy Profile")
                fig_eiip.update_traces(line_color='#2563EB', line_width=1.8)
                fig_eiip.update_layout(height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_eiip, use_container_width=True)
                
            with sig_tab2:
                st.markdown("##### Fast Fourier Transform (FFT) Power Spectral Density")
                st.caption("Highlights dominant structural harmonics. The 3-base periodicity peak (f = 1/3 ≈ 0.33) is a hallmark of coding regions.")
                df_fft = pd.DataFrame({'Frequency': fft_features['frequencies'], 'Power': fft_features['power_spectrum']})
                fig_fft = px.line(df_fft, x='Frequency', y='Power', title="Genomic Power Spectrum |X(f)|²")
                fig_fft.add_vline(x=0.333, line_dash="dash", line_color="#EF4444", annotation_text="f=1/3 Periodicity", annotation_position="top right")
                fig_fft.update_traces(line_color='#059669', line_width=1.5)
                fig_fft.update_layout(height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_fft, use_container_width=True)
                st.info(f"**Spectral Metrics**: 3-Base Periodicity Ratio: `{fft_features['periodicity_ratio']:.4f}` | Spectral Centroid: `{fft_features['spectral_centroid']:.4f}` | Spectral Entropy: `{fft_features['spectral_entropy']:.4f}`")
                
            with sig_tab3:
                st.markdown("##### Hilbert Transform Analytic Signal & Instantaneous Envelope")
                st.caption("Calculates the instantaneous amplitude envelope |z[n]|. Abrupt spikes identify localized mutation clusters.")
                df_hilbert = pd.DataFrame({
                    'Position': np.arange(len(hilbert_features['envelope'][:250])),
                    'Instantaneous Envelope': hilbert_features['envelope'][:250],
                    'Instantaneous Frequency': hilbert_features['instantaneous_freq'][:250]
                })
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=df_hilbert['Position'], y=df_hilbert['Instantaneous Envelope'], name="Amplitude Envelope", line=dict(color='#8B5CF6', width=2)))
                fig_h.add_trace(go.Scatter(x=df_hilbert['Position'], y=df_hilbert['Instantaneous Frequency'], name="Instantaneous Frequency", line=dict(color='#F59E0B', width=1, dash='dot')))
                fig_h.update_layout(title="Hilbert Analytic Signal Decomposition (First 250 bps)", height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_h, use_container_width=True)
                st.info(f"**Hilbert Metrics**: Mean Envelope: `{hilbert_features['mean_envelope']:.4f}` | Envelope Variance: `{hilbert_features['envelope_variance']:.4f}` | Crest Factor: `{hilbert_features['crest_factor']:.2f}`")

            with sig_tab4:
                st.markdown("##### K-Mer Sequence Graph Network Topology")
                st.caption("Directed transition network among 3-mer motifs. Nodes = K-mers, size = occurrences, color = node degree.")
                fig_kmer = create_kmer_graph_plot(kmer_graph, max_nodes=35)
                st.plotly_chart(fig_kmer, use_container_width=True)
                km_c1, km_c2, km_c3, km_c4 = st.columns(4)
                km_c1.metric("Unique K-Mers", int(kmer_metrics['num_nodes']))
                km_c2.metric("Transitions (Edges)", int(kmer_metrics['num_edges']))
                km_c3.metric("Graph Density", kmer_metrics['graph_density'])
                km_c4.metric("Transition Entropy", kmer_metrics['transition_entropy'])
                
            with sig_tab5:
                st.markdown("##### PyTorch BiLSTM Self-Attention Hotspot Localization")
                st.caption("Highlights specific nucleotide positions that contributed highest attention weights to the classification decision.")
                attn_weights = pred_result['attention_weights']
                df_attn = pd.DataFrame({'Position': np.arange(len(attn_weights)), 'Attention Weight': attn_weights})
                fig_attn = px.area(df_attn, x='Position', y='Attention Weight', title="Genomic Attention Weight Profile Across Sequence Loci")
                fig_attn.update_traces(line_color='#DC2626' if is_cancer else '#059669', fillcolor='rgba(239, 68, 68, 0.2)' if is_cancer else 'rgba(16, 185, 129, 0.2)')
                fig_attn.update_layout(height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_attn, use_container_width=True)

            # Clinical Report Export
            report_text = f"""======================================================================
AI CLINICAL GENOMIC SEQUENCING REPORT
AI-Based Early Lung Cancer Detection and Classification from Genomic Signals
Department of Computer Science & Engineering, Siddaganga Institute of Technology
======================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Diagnostic Pipeline: EIIP -> FFT & Hilbert [K-mer Graph] -> PyTorch BiLSTM
----------------------------------------------------------------------
SEQUENCE CHARACTERISTICS:
- Total Base Length: {base_comp['total_length']} bps
- GC Content: {base_comp['gc_content']}%
- Nucleotide Frequencies: A={base_comp['counts']['A']} ({base_comp['percentages']['A']}%), C={base_comp['counts']['C']} ({base_comp['percentages']['C']}%), G={base_comp['counts']['G']} ({base_comp['percentages']['G']}%), T={base_comp['counts']['T']} ({base_comp['percentages']['T']}%)
- Leading Strand (50 bps): {clean_seq[:50]}...
----------------------------------------------------------------------
PREDICTION & MOLECULAR STRATIFICATION:
- Clinical Diagnosis: {pred_label}
- BiLSTM Confidence Score: {pred_result['confidence_score']}%
- Cancer Probability: {pred_result['cancer_probability']}%
- Normal Probability: {pred_result['normal_probability']}%
- Mutational Burden (TMB): {pred_result['mutation_burden']}%
- Mutation Risk Level: {risk_level}
----------------------------------------------------------------------
SPECTRAL & TOPOLOGICAL METRICS:
- FFT 3-Base Periodicity: {fft_features['periodicity_ratio']:.4f}
- Spectral Centroid: {fft_features['spectral_centroid']:.4f}
- Hilbert Mean Envelope: {hilbert_features['mean_envelope']:.4f}
- K-Mer Graph Transition Entropy: {kmer_metrics['transition_entropy']}
======================================================================
DISCLAIMER: For academic and investigative bioinformatics research purposes.
"""
            st.download_button(
                label="📄 Export Clinical Report (.txt)",
                data=report_text,
                file_name=f"Clinical_Genomic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# -------------------------------------------------------------
# PAGE 3: UPLOAD DNA FILE
# -------------------------------------------------------------
elif selected_page == "Upload DNA File":
    st.markdown("### Upload DNA Sequence File")
    st.write("Upload a `.fasta`, `.txt`, or `.csv` file containing a DNA sequence for classification using the PyTorch BiLSTM engine.")
    
    uploaded_file = st.file_uploader("Upload genomic sequence file", type=['fasta', 'txt', 'csv'])
    
    if uploaded_file is not None:
        try:
            records = []
            filename = uploaded_file.name
            
            if filename.endswith('.fasta') or filename.endswith('.txt'):
                content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                lines = content.splitlines()
                current_id = "SEQ_001"
                current_seq = []
                
                for line in lines:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_seq:
                            records.append({'id': current_id, 'sequence': "".join(current_seq)})
                            current_seq = []
                        current_id = line[1:].split()[0]
                    elif line:
                        current_seq.append(line)
                        
                if current_seq:
                    records.append({'id': current_id, 'sequence': "".join(current_seq)})
                    
            elif filename.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
                
                seq_col = None
                for col_name in ['sequence', 'dna_sequence', 'seq']:
                    if col_name in df_upload.columns:
                        seq_col = col_name
                        break
                if seq_col is None:
                    seq_col = df_upload.columns[1] if len(df_upload.columns) > 1 else df_upload.columns[0]
                
                id_col = None
                for col_name in ['id', 'sequence_id', 'sample_id']:
                    if col_name in df_upload.columns:
                        id_col = col_name
                        break
                
                for i, row in df_upload.iterrows():
                    seq_id = str(row[id_col]) if id_col else f"SEQ_{i+1:03d}"
                    records.append({'id': seq_id, 'sequence': str(row[seq_col])})
                    
            st.success(f"Successfully parsed `{filename}`: Loaded **{len(records)}** sequence(s).")
            
            # Use first sequence for detailed prediction (same style as DNA Prediction page)
            if records:
                rec = records[0]
                seq = clean_dna_sequence(rec['sequence'])
                
                if not seq or not all(c in 'ACGT' for c in seq):
                    st.error("❌ Invalid DNA sequence. Only A, C, G, T characters are allowed.")
                elif not has_genomic_model:
                    st.error(f"❌ Deep Learning Model Unavailable: {model_status}. Please ensure training has completed.")
                else:
                    if st.button("Run Inference 🚀", type="primary"):
                        with st.spinner("Processing EIIP encoding, FFT, Hilbert transform, K-mer graph, and LSTM inference..."):
                            eiip_sig = eiip_encode(seq, target_length=500)
                            fft_features = compute_fft_features(eiip_sig)
                            hilbert_features = compute_hilbert_features(eiip_sig)
                            kmer_graph = build_kmer_graph(seq, k=3)
                            kmer_metrics = compute_kmer_graph_metrics(kmer_graph)
                            feat_matrix = extract_combined_signal_matrix(eiip_sig, target_length=500)
                            thresh = metadata['test_metrics'].get('optimal_threshold', 0.50) if metadata else 0.50
                            pred_result = predict_single_sequence(genomic_model, feat_matrix, threshold=thresh)
                        
                        # Temporary Debug Logging (Requirement 14)
                        print(f"[DEBUG INFERENCE UPLOAD] First 30 bp: {seq[:30]} | Length: {len(seq)} | Feature matrix: {feat_matrix.shape} | Normal prob: {pred_result['normal_probability']}% | Cancer prob: {pred_result['cancer_probability']}%", flush=True)
                        with st.expander("🛠️ Inference Debug Diagnostics (Active Run)", expanded=False):
                            d_c1, d_c2, d_c3 = st.columns(3)
                            d_c1.write(f"**Cleaned Seq (First 30 bp)**: `{seq[:30]}`")
                            d_c1.write(f"**Sequence Length**: `{len(seq)} bp`")
                            d_c2.write(f"**Feature Matrix Shape**: `{feat_matrix.shape}`")
                            d_c2.write(f"**Tensor Input Shape**: `(1, {feat_matrix.shape[0]}, {feat_matrix.shape[1]})`")
                            d_c3.write(f"**Normal Probability**: `{pred_result['normal_probability']}%`")
                            d_c3.write(f"**Cancer Probability**: `{pred_result['cancer_probability']}%`")
                            
                        # Display Prediction Banner Cards (same as DNA Prediction page)
                        is_cancer = pred_result['is_cancer']
                        pred_label = "Cancer" if is_cancer else "Normal"
                        pred_color = "#EF4444" if is_cancer else "#10B981"
                        risk_level = "High Risk" if is_cancer and pred_result['mutation_burden'] > 30 else ("Moderate Risk" if is_cancer else "Low / Baseline")
                        risk_color = "#EF4444" if "High" in risk_level else ("#F59E0B" if "Moderate" in risk_level else "#10B981")
                        
                        st.markdown("---")
                        st.markdown("### 🔬 Molecular Analysis & Clinical Inference")
                        
                        res_c1, res_c2, res_c3 = st.columns(3)
                        with res_c1:
                            st.markdown(f'<div class="card" style="border-top: 5px solid {pred_color};"><div class="stat-label">Model Diagnosis</div><div class="stat-value" style="color: {pred_color}; font-size: 1.6rem;">{pred_label}</div></div>', unsafe_allow_html=True)
                        with res_c2:
                            st.markdown(f'<div class="card" style="border-top: 5px solid #2563EB;"><div class="stat-label">Prediction Confidence</div><div class="stat-value" style="font-size: 1.6rem;">{pred_result["confidence_score"]}%</div></div>', unsafe_allow_html=True)
                        with res_c3:
                            st.markdown(f'<div class="card" style="border-top: 5px solid {risk_color};"><div class="stat-label">Mutational Burden (TMB)</div><div class="stat-value" style="color: {risk_color}; font-size: 1.6rem;">{pred_result["mutation_burden"]}%</div></div>', unsafe_allow_html=True)
                        
                        # Base composition & viewer
                        base_comp = compute_base_composition(seq)
                        st.markdown("#### DNA Sequence Base Viewer")
                        st.markdown(render_dna_sequence(seq[:800] + ("..." if len(seq) > 800 else "")), unsafe_allow_html=True)
                        st.caption(f"Sequence Length: **{base_comp['total_length']} bps** | GC Content: **{base_comp['gc_content']}%** | A: {base_comp['percentages']['A']}% | C: {base_comp['percentages']['C']}% | G: {base_comp['percentages']['G']}% | T: {base_comp['percentages']['T']}%")
                        
                        # Probability Distribution & Mutation Gauge
                        sub_col1, sub_col2 = st.columns([1.2, 1])
                        with sub_col1:
                            st.markdown("#### Binary Class Probability Distribution")
                            df_prob = pd.DataFrame([
                                {'Class': 'Cancer', 'Probability (%)': pred_result['cancer_probability']},
                                {'Class': 'Normal', 'Probability (%)': pred_result['normal_probability']}
                            ])
                            fig_prob = px.bar(
                                df_prob, x='Probability (%)', y='Class', orientation='h',
                                color='Class',
                                color_discrete_map={'Cancer': '#EF4444', 'Normal': '#10B981'},
                                text='Probability (%)'
                            )
                            fig_prob.update_layout(height=260, showlegend=False, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_prob, use_container_width=True)
                            
                        with sub_col2:
                            st.markdown("#### Mutational Burden Stratification")
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=pred_result['mutation_burden'],
                                number={'suffix': "%"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': risk_color},
                                    'steps': [
                                        {'range': [0, 10], 'color': "rgba(16, 185, 129, 0.2)"},
                                        {'range': [10, 30], 'color': "rgba(245, 158, 11, 0.2)"},
                                        {'range': [30, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                                    ]
                                }
                            ))
                            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Signal Visualizations
                        st.markdown("---")
                        st.markdown("### 📊 Signal Processing & Deep Feature Representations")
                        sig_tab1, sig_tab2, sig_tab3, sig_tab4, sig_tab5 = st.tabs([
                            "1. EIIP Signal",
                            "2. FFT Power Spectrum",
                            "3. Hilbert Envelope",
                            "4. K-Mer Transition Graph",
                            "5. LSTM Attention Hotspots"
                        ])
                        
                        with sig_tab1:
                            st.markdown("##### Electron-Ion Interaction Pseudopotential (EIIP) 1D Genomic Signal")
                            df_eiip = pd.DataFrame({'Position (bps)': np.arange(len(eiip_sig[:250])), 'EIIP Energy': eiip_sig[:250]})
                            fig_eiip = px.line(df_eiip, x='Position (bps)', y='EIIP Energy', line_shape='hv', title="EIIP Discrete Energy Profile (First 250 bps)")
                            fig_eiip.update_traces(line_color='#2563EB', line_width=1.8)
                            fig_eiip.update_layout(height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_eiip, use_container_width=True)
                            
                        with sig_tab2:
                            st.markdown("##### Fast Fourier Transform (FFT) Power Spectral Density")
                            df_fft = pd.DataFrame({'Frequency': fft_features['frequencies'], 'Power': fft_features['power_spectrum']})
                            fig_fft = px.line(df_fft, x='Frequency', y='Power', title="Genomic Power Spectrum |X(f)|²")
                            fig_fft.add_vline(x=0.333, line_dash="dash", line_color="#EF4444", annotation_text="f=1/3 Periodicity", annotation_position="top right")
                            fig_fft.update_traces(line_color='#059669', line_width=1.5)
                            fig_fft.update_layout(height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_fft, use_container_width=True)
                            
                        with sig_tab3:
                            st.markdown("##### Hilbert Transform Analytic Signal & Instantaneous Envelope")
                            df_hilbert = pd.DataFrame({
                                'Position': np.arange(len(hilbert_features['envelope'][:250])),
                                'Instantaneous Envelope': hilbert_features['envelope'][:250],
                                'Instantaneous Frequency': hilbert_features['instantaneous_freq'][:250]
                            })
                            fig_h = go.Figure()
                            fig_h.add_trace(go.Scatter(x=df_hilbert['Position'], y=df_hilbert['Instantaneous Envelope'], name="Amplitude Envelope", line=dict(color='#8B5CF6', width=2)))
                            fig_h.add_trace(go.Scatter(x=df_hilbert['Position'], y=df_hilbert['Instantaneous Frequency'], name="Instantaneous Frequency", line=dict(color='#F59E0B', width=1, dash='dot')))
                            fig_h.update_layout(title="Hilbert Analytic Signal Decomposition (First 250 bps)", height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_h, use_container_width=True)
                        
                        with sig_tab4:
                            st.markdown("##### K-Mer Sequence Graph Network Topology")
                            fig_kmer = create_kmer_graph_plot(kmer_graph, max_nodes=35)
                            st.plotly_chart(fig_kmer, use_container_width=True)
                            
                        with sig_tab5:
                            st.markdown("##### PyTorch BiLSTM Self-Attention Hotspot Localization")
                            attn_weights = pred_result['attention_weights']
                            df_attn = pd.DataFrame({'Position': np.arange(len(attn_weights)), 'Attention Weight': attn_weights})
                            fig_attn = px.area(df_attn, x='Position', y='Attention Weight', title="Genomic Attention Weight Profile Across Sequence Loci")
                            fig_attn.update_traces(line_color='#DC2626' if is_cancer else '#059669', fillcolor='rgba(239, 68, 68, 0.2)' if is_cancer else 'rgba(16, 185, 129, 0.2)')
                            fig_attn.update_layout(height=320, plot_bgcolor='rgba(248,250,252,0.6)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_attn, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------------------------------------
# PAGE 4: ANALYTICS & MODEL DIAGNOSTICS
# -------------------------------------------------------------
elif selected_page == "Analytics":
    st.markdown("### Model Diagnostics & Validation Analytics")
    st.write("Rigorous quantitative evaluation of the PyTorch BiLSTM Genomic Classifier on held-out test data.")
    
    if metadata:
        test_m = metadata['test_metrics']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation Accuracy", f"{test_m['accuracy']}%")
        c2.metric("Precision", f"{test_m['precision']}%")
        c3.metric("Recall (Sensitivity)", f"{test_m['recall']}%")
        c4.metric("ROC-AUC", f"{test_m['roc_auc']}")
        
        st.markdown("---")
        
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
            fpr = metadata['roc_curve']['fpr']
            tpr = metadata['roc_curve']['tpr']
            fig_roc = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC = {test_m['roc_auc']})",
                              labels={'x': 'False Positive Rate (1 - Specificity)', 'y': 'True Positive Rate (Sensitivity)'})
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_traces(line_color='#2563EB', line_width=2.5)
            fig_roc.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with row1_c2:
            st.markdown("#### Test Set Confusion Matrix")
            cm = np.array(metadata['confusion_matrix'])
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale='Blues',
                labels=dict(x="Predicted Label", y="Ground Truth", color="Sequences"),
                x=['Normal (-ve)', 'Cancer (+ve)'],
                y=['Normal (-ve)', 'Cancer (+ve)'],
                title=f"Binary Confusion Matrix"
            )
            fig_cm.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm, use_container_width=True)
            
        st.markdown("#### Training & Validation Convergence Curves")
        hist = metadata['training_history']
        epochs = np.arange(1, len(hist['train_loss']) + 1)
        df_hist = pd.DataFrame({
            'Epoch': epochs,
            'Train Loss': hist['train_loss'],
            'Validation Loss': hist['val_loss'],
            'Validation Accuracy': [acc * 100 for acc in hist['val_acc']]
        })
        
        loss_col, acc_col = st.columns(2)
        with loss_col:
            fig_loss = px.line(df_hist, x='Epoch', y=['Train Loss', 'Validation Loss'],
                               title="Training & Validation Cross-Entropy Loss", markers=True)
            fig_loss.update_layout(height=340, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_loss, use_container_width=True)
        with acc_col:
            fig_acc = px.line(df_hist, x='Epoch', y='Validation Accuracy',
                              title="Validation Accuracy Progression (%)", markers=True)
            fig_acc.update_traces(line_color='#10B981', line_width=2)
            fig_acc.update_layout(height=340, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.warning("Model metadata file not found. Please execute train.py to generate analytics.")

# -------------------------------------------------------------
# PAGE 5: COMPARATIVE STUDY
# -------------------------------------------------------------
elif selected_page == "Comparative Study":
    st.markdown("### Comparative Study: Genomic Signals vs PET/CT Multi-Modal Foundation Model")
    st.write(
        "A systematic cross-modality comparative study between our AI model trained on **1D Genomic Signals (EIIP + BiLSTM)** "
        "versus the state-of-the-art **PET/CT Foundation Model (FratMAE)** for cross-modal anatomical and functional imaging "
        "([arXiv:2503.02824](https://arxiv.org/html/2503.02824v1))."
    )
    
    st.markdown("""
    <div class="card" style="border-left: 5px solid #2563EB; background-color: #F8FAFC; margin-bottom: 20px;">
        <h4 style="color: #1E3A8A; margin-top: 0; margin-bottom: 6px;">📄 Reference Foundation Model Benchmark</h4>
        <p style="margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 600; color: #0F172A;">
            <a href="https://arxiv.org/html/2503.02824v1" target="_blank" style="color: #2563EB; text-decoration: underline;">
                Developing a PET/CT Foundation Model for Cross-Modal Anatomical and Functional Imaging
            </a>
        </p>
        <p style="margin: 0 0 4px 0; font-size: 0.9rem; color: #475569;">
            <b>Authors:</b> Yujin Oh, Robert Seifert, Yihan Cao, Quanzheng Li, Ning Guo, et al. (Harvard Medical School, MGH, Univ. of Bern, Cologne, Augsburg, ShanghaiTech)
        </p>
        <p style="margin: 0; font-size: 0.88rem; color: #64748B;">
            <b>Key Innovation:</b> <i>FratMAE (Cross-Fraternal Twin Masked Autoencoder)</i> with dual 3D Vision Transformers & cross-attention decoders, unifying anatomical CT and metabolic <sup>18</sup>F-FDG PET functional imaging on the multi-center AutoPET cohort.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### PET/CT Foundation Model (FratMAE) Benchmark Highlights")
    
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Staging Accuracy", "65.4%", help="FratMAE multi-class lymphoma staging under low-label/few-shot constraints")
    f2.metric("Staging Sensitivity", "64.8%", help="True positive rate for clinical staging")
    f3.metric("Staging Specificity", "71.2%", help="True negative rate across multi-center validation")
    f4.metric("Staging ROC-AUC", "0.824", help="Multi-class cross-modal discrimination ROC-AUC")
    f5.metric("Segmentation Dice", "64.0%", help="AutoPET 3D whole-body lesion segmentation (IoU: 49.6%)")
    
    st.markdown("---")
    st.markdown("#### Cross-Modality Benchmark Comparison Table")

    matrix_data = get_comparative_matrix()
    dim = matrix_data['dimensions']
    gen = matrix_data['genomic']
    img = matrix_data['imaging']

    comp_df = pd.DataFrame({
        'Diagnostic Dimension': dim,
        'Genomic AI Model (Our Framework)': [
            gen['modality'],
            gen['input'],
            gen['stage'],
            gen['capability'],
            gen['risk'],
            gen['model'],
            gen['subtype'],
            gen['accuracy'],
            gen['sensitivity'],
            gen['specificity'],
            gen['auc'],
            gen['dice'],
            gen['latency'],
            gen['turnaround']
        ],
        'PET/CT Foundation Model (FratMAE - arXiv:2503.02824)': [
            img['modality'],
            img['input'],
            img['stage'],
            img['capability'],
            img['risk'],
            img['model'],
            img['subtype'],
            img['accuracy'],
            img['sensitivity'],
            img['specificity'],
            img['auc'],
            img['dice'],
            img['latency'],
            img['turnaround']
        ]
    })

    st.table(comp_df)


    # ============================================================
    # PERFORMANCE COMPARISON
    # ============================================================

    st.markdown("---")
    st.markdown("#### Performance Comparison: Genomic AI vs PET/CT Foundation Model (FratMAE)")

    st.info(
        "Genomic AI model metrics represent our held-out test evaluation on 1D nucleotide signals. "
        "PET/CT metrics are reported from the peer-reviewed benchmark in 'Developing a PET/CT Foundation Model for Cross-Modal Anatomical and Functional Imaging' (arXiv:2503.02824)."
    )

    # Only plot metrics if valid numerical values are available
    def parse_metric(value):
        try:
            if isinstance(value, str):
                return float(value.replace('%', ''))
            return float(value)
        except:
            return None


    genomic_values = [
        parse_metric(gen['accuracy']),
        parse_metric(gen['sensitivity']),
        parse_metric(gen['specificity']),
        parse_metric(gen['auc']) * 100 if parse_metric(gen['auc']) is not None else None
    ]

    imaging_values = [
        parse_metric(img['accuracy']),
        parse_metric(img['sensitivity']),
        parse_metric(img['specificity']),
        parse_metric(img['auc']) * 100 if parse_metric(img['auc']) is not None else None
    ]

    metric_names = [
        'Accuracy (%)',
        'Sensitivity (%)',
        'Specificity (%)',
        'AUC-ROC (%)'
    ]

    # Plot only when both models have valid numerical values
    if all(v is not None for v in genomic_values) and all(v is not None for v in imaging_values):

        fig_comp = go.Figure()

        fig_comp.add_trace(go.Scatter(
            x=metric_names,
            y=genomic_values,
            mode='lines+markers',
            name='Genomic AI Model (BiLSTM)',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=10, symbol='circle')
        ))

        fig_comp.add_trace(go.Scatter(
            x=metric_names,
            y=imaging_values,
            mode='lines+markers',
            name='PET/CT Foundation Model (FratMAE)',
            line=dict(color='#10B981', width=3),
            marker=dict(size=10, symbol='diamond')
        ))

        fig_comp.update_layout(
            title="Performance Metrics: Genomic Signals vs PET/CT Foundation Model (FratMAE)",
            xaxis_title="Diagnostic / Classification Metric",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 105]),
            height=450,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )

        st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.warning(
            "A direct numerical performance comparison is not displayed because "
            "comparable PET/CT evaluation metrics are not available."
        )


    # ============================================================
    # CONCLUSION
    # ============================================================

    st.markdown("---")
    st.markdown("#### Clinical & Technological Synthesis")

    st.markdown("""
    <div class="card" style="background-color: #F8FAFC; border-left: 5px solid #10B981;">

    <h4>🔬 Genomic Signal Analysis vs. Cross-Modal PET/CT Foundation Imaging: Dual Paradigms in Precision Oncology</h4>

    <ul style="line-height: 1.9; color: #334155;">

    <li>
    <b>Cross-Modality Complementarity:</b>
    Our genomic framework analyzes 1D DNA nucleotide sequence signals (ctDNA liquid biopsy) to detect microscopic oncogenic driver mutations (<i>TP53, EGFR, KRAS</i>). In contrast, the PET/CT foundation model (FratMAE, arXiv:2503.02824) processes paired 3D volumetric scans fusing high-resolution anatomical structures (CT) with functional cellular glucose metabolism (<sup>18</sup>F-FDG PET).
    </li>

    <li>
    <b>Temporal Diagnostic Windows:</b>
    Genomic signal processing operates at the earliest <b>Stage I (pre-symptomatic)</b> window before malignant cells coalesce into macroscopic nodules detectable by radiological scanners. Conversely, PET/CT foundation models excel at <b>Stage II–IV whole-body disease staging</b> and 3D spatial lesion delineation.
    </li>

    <li>
    <b>Functional Hypermetabolism vs. Anatomical Nodules:</b>
    Conventional standalone CT only identifies structural mass and physical size (>3–5 mm), frequently producing false positives from benign granulomas or scars. Dual PET/CT overcomes this by measuring Standardized Uptake Values (SUV), distinguishing active hypermetabolic malignant tissue from necrotic or benign lesions.
    </li>

    <li>
    <b>Deep Learning Architectural Synergy:</b>
    Our model utilizes <b>EIIP physical encoding + FFT power spectral density (f=1/3 codon marker) + Hilbert analytic signal envelope + 2-layer Bidirectional LSTM with Attention pooling</b> for high-sensitivity sequence classification. FratMAE utilizes <b>Cross-Fraternal Twin Vision Transformers (ViT)</b> with cross-attention masked autoencoding and whole-body contextual metadata alignment for self-supervised pre-training on multi-center cohorts (AutoPET).
    </li>

    <li>
    <b>Segmentation vs. Sequence Locus Localization:</b>
    While FratMAE provides 3D voxel-level lesion segmentation (Dice: <b>64.0%</b>, IoU: <b>49.6%</b>), our Genomic AI framework provides nucleotide-level self-attention weight heatmaps pinpointing specific codon mutational hotspots along the DNA strand.
    </li>

    <li>
    <b>Integrated Precision Oncology Pipeline:</b>
    Neither modality supersedes the other; rather, they form the ideal two-tiered diagnostic standard: minimally invasive genomic liquid biopsies serve as population-wide early screening, while PET/CT foundation models provide whole-body tumor localization, spatial staging, and guidance for surgical resection or targeted radiotherapy.
    </li>

    </ul>

    </div>
    """, unsafe_allow_html=True)
# -------------------------------------------------------------
# PAGE 6: TECH STACK
# -------------------------------------------------------------
elif selected_page == "Tech Stack":
    st.markdown("### System Technology Architecture")
    st.write("The framework integrates advanced bioinformatics signal processing, PyTorch deep learning, and interactive data visualization.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card" style="border-top: 5px solid #3B82F6; min-height: 360px;">
        <h4>Presentation & Dashboard Layer</h4>
        <ul style="line-height: 1.9; list-style-type: none; padding-left: 0;">
            <li>🐍 <b>Python 3.11</b>: Core development runtime</li>
            <li>👑 <b>Streamlit 1.49</b>: Reactive healthcare analytical dashboard</li>
            <li>📊 <b>Plotly Graph Objects</b>: Interactive FFT, Hilbert, and K-mer graphs</li>
            <li>🎨 <b>Custom Modern CSS</b>: Responsive typography and card styling</li>
            <li>📑 <b>Clinical Report Exporter</b>: Automated diagnostic report generator</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="border-top: 5px solid #10B981; min-height: 360px;">
        <h4>Genomic AI & Signal Processing</h4>
        <ul style="line-height: 1.9; list-style-type: none; padding-left: 0;">
            <li>⚡ <b>EIIP Encoding</b>: Physical electron-ion pseudopotential mapping</li>
            <li>🌊 <b>Fast Fourier Transform (FFT)</b>: 3-base periodicity & spectral energy</li>
            <li>📈 <b>Hilbert Transform</b>: Analytic envelope & instantaneous phase</li>
            <li>🕸️ <b>NetworkX</b>: K-mer transition graphs and topological metrics</li>
            <li>🧠 <b>PyTorch BiLSTM</b>: 2-layer Bidirectional LSTM with Attention pooling</li>
            <li>📉 <b>Scikit-Learn</b>: Stratified splitting, ROC-AUC, and confusion metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card" style="border-top: 5px solid #F59E0B; min-height: 360px;">
        <h4>Medical Imaging & Data Repositories</h4>
        <ul style="line-height: 1.9; list-style-type: none; padding-left: 0;">
            <li>🖼️ <b>PET/CT Foundation Model (FratMAE)</b>: Cross-modal Vision Transformer (arXiv:2503.02824)</li>
            <li>🏥 <b>AutoPET Benchmark</b>: Multi-center whole-body PET/CT imaging cohorts</li>
            <li>🧬 <b>NCBI RefSeq & COSMIC</b>: Authentic TP53, EGFR, KRAS gene loci</li>
            <li>💾 <b>PyTorch Checkpoints</b>: Serialized weights (genomic_lstm.pth)</li>
            <li>📁 <b>FASTA & CSV Engines</b>: High-throughput batch genomic loaders</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 7: TEAM DETAILS
# -------------------------------------------------------------
elif selected_page == "Team Details":
    st.markdown("### Engineering Team & Institutional Attribution")
    
    st.markdown("""
    <div class="card" style="border-top: 5px solid #1E3A8A; background-color: #F8FAFC; text-align: center;">
        <h2 style="color: #1E3A8A; margin-top: 0; margin-bottom: 5px;">Siddaganga Institute of Technology</h2>
        <p style="color: #475569; margin: 0; font-size: 1.2rem;"><b>Department of Computer Science & Engineering</b></p>
        <p style="color: #64748B; margin-bottom: 12px;">Tumakuru, Karnataka, India</p>
        <hr style="border: 0; height: 1px; background: #CBD5E1; margin: 15px 0;">
        <h4 style="color: #0F172A; margin-bottom: 4px;">Project Title: AI-Based Early Lung Cancer Detection and Classification from Genomic Signals</h4>
        <p style="color: #64748B; font-weight: bold; margin-bottom: 0;">Academic Year: 2025 - 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card" style="min-height: 490px; border-top: 5px solid #3B82F6;">
            <h4>Student Engineering Team</h4>
            <br>
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background-color: #DBEAFE; color: #1E40AF; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1rem; margin-right: 15px;">SH</div>
                <div>
                    <h5 style="margin: 0; font-size: 1.05rem; color: #0F172A;">Shreeya H M</h5>
                    <span style="color: #64748B; font-size: 0.85rem;">USN: <b>1SI23CS167</b></span>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background-color: #D1FAE5; color: #065F46; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1rem; margin-right: 15px;">VU</div>
                <div>
                    <h5 style="margin: 0; font-size: 1.05rem; color: #0F172A;">Vidhyashree U</h5>
                    <span style="color: #64748B; font-size: 0.85rem;">USN: <b>1SI23CS194</b></span>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background-color: #FEF3C7; color: #92400E; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1rem; margin-right: 15px;">SB</div>
                <div>
                    <h5 style="margin: 0; font-size: 1.05rem; color: #0F172A;">Shrinivas Bhagavantagoudar</h5>
                    <span style="color: #64748B; font-size: 0.85rem;">USN: <b>1SI23CS169</b></span>
                </div>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background-color: #F3E8FF; color: #6B21A8; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1rem; margin-right: 15px;">SS</div>
                <div>
                    <h5 style="margin: 0; font-size: 1.05rem; color: #0F172A;">Shrishantha</h5>
                    <span style="color: #64748B; font-size: 0.85rem;">USN: <b>1SI23CS170</b></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card" style="min-height: 490px; border-top: 5px solid #10B981;">
            <h4>Project Guide & Supervision</h4>
            <br>
            <div style="display: flex; align-items: center; margin-bottom: 25px;">
                <div style="width: 58px; height: 58px; border-radius: 50%; background-color: #F1F5F9; color: #334155; border: 2px solid #CBD5E1; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; margin-right: 15px;">👨‍🏫</div>
                <div>
                    <h5 style="margin: 0; font-size: 1.15rem; color: #0F172A;">Mr. Gururaj S P</h5>
                    <span style="color: #64748B; font-size: 0.95rem;">Assistant Professor, Department of CSE</span>
                </div>
            </div>
            <hr style="border: 0; height: 1px; background: #E2E8F0; margin: 15px 0;">
            <p style="color: #334155; font-size: 0.95rem;">📅 <b>Academic Session</b>: Major Project (Phase 3)</p>
            <p style="color: #334155; font-size: 0.95rem;">🚀 <b>Core Domain</b>: Computational Biology, Deep Learning & Medical Imaging</p>
            <p style="color: #334155; font-size: 0.95rem;">🔬 <b>Methodology</b>: EIIP Numerical Encoding, FFT Spectral Periodicity, Hilbert Transform, PyTorch BiLSTM Attention Network</p>
            <p style="color: #334155; font-size: 0.95rem;">💻 <b>Benchmarking</b>: PET/CT Foundation Model for Anatomical & Functional Imaging (FratMAE, arXiv:2503.02824)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: #94A3B8; font-style: italic; margin-top: 30px; margin-bottom: 20px;'>Dedicated to advancing AI-driven healthcare diagnostics and precision oncology.</p>", unsafe_allow_html=True)

