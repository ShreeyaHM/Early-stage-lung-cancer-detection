"""
Genomic Signal Processing: FFT & Hilbert Transform Feature Extraction
Extracts frequency spectrum characteristics, 3-base periodicity, and instantaneous
envelope dynamics from EIIP numerical genomic signals.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from typing import Dict, Any, Tuple

def compute_fft_features(signal: np.ndarray) -> Dict[str, Any]:
    """
    Computes Fast Fourier Transform (FFT) and spectral features of the genomic signal.
    
    In genomic signal processing, protein-coding regions exhibit a prominent peak
    at normalized frequency f = 1/3 (3-base periodicity). Somatic mutations and deletions
    distort this spectral energy profile.
    
    Parameters:
        signal (np.ndarray): 1D float array of EIIP values.
        
    Returns:
        dict containing:
            - frequencies: array of discrete frequencies
            - power_spectrum: power spectral density (|X[k]|^2 / N)
            - periodicity_ratio: ratio of energy around f=1/3 to total energy
            - spectral_centroid: center of mass of spectrum
            - spectral_energy: total sum of spectral power
            - peak_frequency: dominant frequency in spectrum
    """
    n = len(signal)
    if n < 4:
        return {
            'frequencies': np.array([0.0]),
            'power_spectrum': np.array([0.0]),
            'periodicity_ratio': 0.0,
            'spectral_centroid': 0.0,
            'spectral_energy': 0.0,
            'peak_frequency': 0.0,
            'spectral_entropy': 0.0
        }
    
    # Detrend by removing DC mean component
    detrended = signal - np.mean(signal)
    
    # Compute FFT
    X = fft(detrended)
    power = (np.abs(X) ** 2) / n
    freqs = fftfreq(n, d=1.0)
    
    # Keep only positive frequencies (single-sided spectrum)
    half_idx = n // 2
    pos_freqs = freqs[:half_idx]
    pos_power = power[:half_idx]
    
    total_energy = np.sum(pos_power) + 1e-12
    
    # Periodicity at f = 1/3 (index closest to 1/3)
    target_freq = 1.0 / 3.0
    periodicity_idx = np.argmin(np.abs(pos_freqs - target_freq))
    
    # Window around 1/3 (3 bins)
    win_start = max(0, periodicity_idx - 1)
    win_end = min(half_idx, periodicity_idx + 2)
    periodicity_energy = np.sum(pos_power[win_start:win_end])
    periodicity_ratio = float(periodicity_energy / total_energy)
    
    # Spectral centroid
    spectral_centroid = float(np.sum(pos_freqs * pos_power) / total_energy)
    
    # Peak frequency (excluding DC if near zero)
    if half_idx > 1:
        peak_idx = np.argmax(pos_power[1:]) + 1
        peak_freq = float(pos_freqs[peak_idx])
    else:
        peak_freq = 0.0
        
    # Spectral entropy
    prob = pos_power / total_energy
    prob = prob[prob > 0]
    spectral_entropy = float(-np.sum(prob * np.log2(prob)))
    
    return {
        'frequencies': pos_freqs,
        'power_spectrum': pos_power,
        'periodicity_ratio': periodicity_ratio,
        'spectral_centroid': spectral_centroid,
        'spectral_energy': float(total_energy),
        'peak_frequency': peak_freq,
        'spectral_entropy': spectral_entropy
    }

def compute_hilbert_features(signal: np.ndarray) -> Dict[str, Any]:
    """
    Computes Hilbert Transform to extract analytic signal, instantaneous amplitude envelope,
    and instantaneous frequency fluctuations.
    
    Somatic mutations introduce abrupt local disruptions in nucleotide order,
    producing sharp spikes in the instantaneous envelope and phase derivatives.
    
    Parameters:
        signal (np.ndarray): 1D float array of EIIP values.
        
    Returns:
        dict containing:
            - envelope: instantaneous amplitude |z[n]|
            - instantaneous_phase: unwrapped phase of analytic signal
            - instantaneous_freq: derivative of unwrapped phase
            - mean_envelope: average envelope amplitude
            - envelope_variance: variance of the envelope
            - crest_factor: peak-to-average ratio
            - envelope_entropy: complexity measure of envelope fluctuations
    """
    n = len(signal)
    if n < 4:
        zeros = np.zeros(max(n, 1), dtype=np.float32)
        return {
            'envelope': zeros,
            'instantaneous_phase': zeros,
            'instantaneous_freq': zeros,
            'mean_envelope': 0.0,
            'envelope_variance': 0.0,
            'crest_factor': 0.0,
            'envelope_entropy': 0.0
        }
        
    # Detrend
    detrended = signal - np.mean(signal)
    
    # Analytic signal z[n] = x[n] + j * H{x[n]}
    analytic_signal = hilbert(detrended)
    
    # Instantaneous amplitude envelope
    envelope = np.abs(analytic_signal).astype(np.float32)
    
    # Instantaneous phase & frequency
    unwrapped_phase = np.unwrap(np.angle(analytic_signal))
    # Normalized instantaneous frequency
    inst_freq = (np.diff(unwrapped_phase) / (2.0 * np.pi))
    inst_freq = np.pad(inst_freq, (0, 1), mode='edge').astype(np.float32)
    
    # Statistical measures
    mean_env = float(np.mean(envelope))
    var_env = float(np.var(envelope))
    max_env = float(np.max(envelope))
    crest_factor = float(max_env / (mean_env + 1e-12))
    
    # Envelope entropy
    hist, _ = np.histogram(envelope, bins=10, density=True)
    hist = hist[hist > 0]
    envelope_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
    
    return {
        'envelope': envelope,
        'instantaneous_phase': unwrapped_phase.astype(np.float32),
        'instantaneous_freq': inst_freq,
        'mean_envelope': mean_env,
        'envelope_variance': var_env,
        'crest_factor': crest_factor,
        'envelope_entropy': envelope_entropy
    }

def extract_combined_signal_matrix(signal: np.ndarray, target_length: int = 500) -> np.ndarray:
    """
    Combines raw EIIP signal, Hilbert instantaneous envelope, and instantaneous frequency
    into a multi-channel sequential matrix shape: (target_length, 3) ready for the LSTM layer.
    The low‑variance local‑energy channel has been removed to match the model's expected input dimension.
    """
    n = len(signal)
    if n == 0:
        return np.zeros((target_length, 3), dtype=np.float32)

    # Truncate or pad signal to target_length
    if n > target_length:
        sig = signal[:target_length]
    else:
        sig = np.pad(signal, (0, target_length - n), mode='constant', constant_values=0.0)

    # Compute Hilbert features
    hilbert_res = compute_hilbert_features(sig)
    envelope = hilbert_res['envelope']
    inst_freq = hilbert_res['instantaneous_freq']

    # Stack features: Channel 0 = EIIP, Channel 1 = Hilbert Envelope, Channel 2 = Instantaneous Frequency
    feature_matrix = np.column_stack([
        sig,
        envelope,
        inst_freq
    ]).astype(np.float32)
    return feature_matrix


def extract_global_feature_vector(sequence: str, eiip_signal: np.ndarray) -> np.ndarray:
    """
    Extracts an 8-dimensional global statistical vector combining FFT, Hilbert, and K-mer graph metrics:
    [periodicity_ratio, spectral_centroid, spectral_entropy, mean_envelope, crest_factor, envelope_entropy, graph_density, transition_entropy]
    """
    from src.kmer_graph import build_kmer_graph, compute_kmer_graph_metrics
    
    fft_f = compute_fft_features(eiip_signal)
    hil_f = compute_hilbert_features(eiip_signal)
    
    clean_seq = "".join(sequence.split()).upper()
    g = build_kmer_graph(clean_seq, k=3)
    km_f = compute_kmer_graph_metrics(g)
    
    vec = np.array([
        fft_f['periodicity_ratio'],
        fft_f['spectral_centroid'],
        fft_f['spectral_entropy'] / 10.0,
        hil_f['mean_envelope'],
        hil_f['crest_factor'] / 10.0,
        hil_f['envelope_entropy'] / 5.0,
        km_f['graph_density'],
        km_f['transition_entropy'] / 5.0
    ], dtype=np.float32)
    
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
