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
    Enhanced CNN-Bidirectional LSTM Deep Learning Model for Binary Genomic Signal Classification.
    
    Pipeline:
      (EIIP, Hilbert Envelope, Inst Freq)
      -> In-model Channel Normalization
      -> Multi-scale 1D Convolutional Motif Detector (kernels 3, 5, 7, 11)
      -> MaxPool1d (2x temporal reduction & local invariance)
      -> 2-Layer Bidirectional LSTM (hidden_size=128, bidirectional=256)
      -> Dual Pooling: Genomic Self-Attention Pooling + Max Pooling + Macro Signal Skip
      -> LayerNorm & GELU Dense Classification Head
      -> Binary Classification Head (Normal -ve vs Cancer Positive +ve)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.20):
        super(GenomicLSTMClassifier, self).__init__()
        
        # Multi-scale 1D Convolutional filters for triplet codons (3), mutations (5, 7), and domains (11)
        self.conv3 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=11, padding=5)
        self.norm_conv = nn.GroupNorm(num_groups=8, num_channels=128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_dim * 2 # bidirectional = 256
        self.attention = GenomicAttention(lstm_out_dim)
        
        # Combined feature size: Attention pooled (256) + Max pooled (256) + Macro Stats (12) = 524
        combined_dim = lstm_out_dim * 2 + (input_dim * 4)
        
        # Dense feature aggregator with LayerNorm for stable train/eval consistency
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )
        
        # Binary Classification Head: 0 = Normal (-ve), 1 = Cancer Positive (+ve)
        self.classifier = nn.Linear(64, 2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        x: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, 2)
            attn_weights: (batch_size, seq_len)
        """
        # In-model sequence-wise channel normalization (zero-mean, unit-variance)
        seq_mean = x.mean(dim=1, keepdim=True)
        seq_std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - seq_mean) / seq_std
        
        # Extract macro sequence statistics
        mean_stat = torch.mean(x_norm, dim=1)
        std_stat = torch.std(x_norm, dim=1)
        max_stat, _ = torch.max(x_norm, dim=1)
        min_stat, _ = torch.min(x_norm, dim=1)
        stats = torch.cat([mean_stat, std_stat, max_stat, min_stat], dim=-1) # (batch, 12)
        
        # Multi-scale Convolutions
        xt = x_norm.transpose(1, 2)
        c3 = F.gelu(self.conv3(xt))
        c5 = F.gelu(self.conv5(xt))
        c7 = F.gelu(self.conv7(xt))
        c11 = F.gelu(self.conv11(xt))
        c = torch.cat([c3, c5, c7, c11], dim=1)
        c = self.norm_conv(c)
        c = self.pool(c) # (batch, 128, seq_len // 2)
        
        # BiLSTM
        lstm_in = c.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        
        # Dual Pooling
        attn_pooled, attn_weights = self.attention(lstm_out)
        max_pooled, _ = torch.max(lstm_out, dim=1)
        
        # Feature fusion and classification
        fused = torch.cat([attn_pooled, max_pooled, stats], dim=-1)
        feat = self.fc(fused)
        logits = self.classifier(feat)
        
        # Rescale attention weights back to original sequence length for nucleotide hotspot visualization
        full_attn = F.interpolate(attn_weights.unsqueeze(1), size=x.size(1), mode='linear', align_corners=False).squeeze(1)
        return logits, full_attn

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
