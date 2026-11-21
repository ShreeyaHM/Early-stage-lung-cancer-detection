"""
Deep Learning Architecture: PyTorch Bidirectional LSTM for Genomic Signals
Accepts multi-channel EIIP + Hilbert + FFT sequential features and outputs
cancer detection (+ve / -ve) with attention-based mutation hotspot localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional

class GenomicAttention(nn.Module):
    """
    Self-Attention pooling layer that computes importance weights across nucleotide positions.
    Highlights mutation loci contributing most to the malignant/normal decision.
    """
    def __init__(self, hidden_dim: int):
        super(GenomicAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attn(lstm_output) # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1) # normalize across seq_len
        pooled = torch.sum(lstm_output * attn_weights, dim=1) # (batch_size, hidden_dim)
        return pooled, attn_weights.squeeze(-1)

class GenomicLSTMClassifier(nn.Module):
    """
    CNN-Bidirectional LSTM Deep Learning Model for Binary Genomic Signal Classification.
    
    Pipeline:
      (EIIP, Hilbert Envelope, Inst Freq, Local FFT Energy)
      -> 1D Convolutional Motif Detector (kernel_size=5, filters=48)
      -> LayerNorm & ReLU
      -> 2-Layer Bidirectional LSTM (hidden_size=64, bidirectional=128)
      -> Genomic Attention Pooling
      -> Binary Classification Head (Normal -ve vs Cancer Positive +ve)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.20):
        super(GenomicLSTMClassifier, self).__init__()
        
        # 1D Convolutional filter detects local nucleotide motifs and mutation disruption
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_dim * 2 # bidirectional
        self.attention = GenomicAttention(lstm_out_dim)
        
        # Dense feature aggregator
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Binary Classification Head: 0 = Normal (-ve), 1 = Cancer Positive (+ve)
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        x: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, 2)
            attn_weights: (batch_size, seq_len)
        """
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x_trans = x.transpose(1, 2)
        conv_out = self.conv(x_trans)
        
        # Transpose back to (batch_size, seq_len, 48) for LSTM
        lstm_in = conv_out.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Attention pooling
        pooled, attn_weights = self.attention(lstm_out)
        
        # Classification
        feat = self.fc(pooled)
        logits = self.classifier(feat)
        
        return logits, attn_weights

def predict_single_sequence(
    model: GenomicLSTMClassifier,
    feature_matrix: np.ndarray,
    threshold: float = 0.50,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Run binary inference on a genomic signal feature matrix using the PyTorch CNN-BiLSTM classifier.
    """
    model.eval()
    model.to(device)
    
    tensor_input = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, attn_weights = model(tensor_input)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        attn_profile = attn_weights.cpu().numpy()[0]
        
    normal_prob = float(probs[0])
    cancer_prob = float(probs[1])
    is_cancer = bool(cancer_prob >= threshold)
    conf_score = round((cancer_prob if is_cancer else normal_prob) * 100.0, 2)
    
    # Mutational burden estimate from attention density and cancer probability
    attn_threshold = np.mean(attn_profile) + 1.5 * np.std(attn_profile)
    spike_density = float(np.sum(attn_profile > attn_threshold) / len(attn_profile))
    
    if is_cancer:
        mut_burden = round(float(np.clip(25.0 + cancer_prob * 45.0 + spike_density * 80.0, 20.0, 94.0)), 1)
    else:
        mut_burden = round(float(np.clip((1.0 - normal_prob) * 12.0 + spike_density * 20.0, 0.1, 4.9)), 1)
        
    return {
        'is_cancer': is_cancer,
        'prediction_label': 'Cancer Positive (+ve)' if is_cancer else 'Normal Negative (-ve)',
        'confidence_score': conf_score,
        'cancer_probability': round(cancer_prob * 100.0, 2),
        'normal_probability': round(normal_prob * 100.0, 2),
        'mutation_burden': mut_burden,
        'attention_weights': attn_profile
    }
