"""
Genomic Signal Processing: Electron-Ion Interaction Pseudopotential (EIIP) Encoding
Maps nucleotide bases (A, C, G, T) into characteristic physical-chemical energy values.
"""

import numpy as np
from typing import Union, List, Tuple

# Standard EIIP values (in Rydberg units) for the four nucleotide bases
EIIP_VALUES = {
    'A': 0.1260,
    'C': 0.1340,
    'G': 0.0806,
    'T': 0.1335,
    'U': 0.1335  # RNA equivalent
}

# Average background EIIP for ambiguous / padding bases
EIIP_MEAN = float(np.mean(list(EIIP_VALUES.values())[:4]))

def clean_dna_sequence(sequence: str) -> str:
    """
    Sanitize and clean raw DNA sequence by removing whitespace, line breaks,
    and converting to uppercase.
    """
    if not sequence:
        return ""
    cleaned = "".join(sequence.split()).upper()
    return cleaned

def validate_dna_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Check if sequence contains only standard nucleotides (A, C, G, T).
    Returns (is_valid, error_message).
    """
    cleaned = clean_dna_sequence(sequence)
    if not cleaned:
        return False, "Sequence is empty."
    
    invalid_chars = set(cleaned) - set('ACGT')
    if invalid_chars:
        return False, f"Sequence contains invalid base characters: {', '.join(sorted(invalid_chars))}. Only A, C, G, T are permitted."
    
    return True, "Valid DNA sequence."

def eiip_encode(sequence: str, target_length: int = None) -> np.ndarray:
    """
    Convert a DNA sequence to a 1D real-valued numerical signal using EIIP values.
    If sequence is shorter than target_length, it is naturally tiled/extended to preserve
    periodic signal properties.
    """
    cleaned = clean_dna_sequence(sequence)
    if not cleaned:
        if target_length:
            return np.full(target_length, EIIP_MEAN, dtype=np.float32)
        return np.array([], dtype=np.float32)
    
    if target_length is not None:
        if len(cleaned) < target_length:
            reps = (target_length // len(cleaned)) + 1
            cleaned = (cleaned * reps)[:target_length]
        elif len(cleaned) > target_length:
            cleaned = cleaned[:target_length]
            
    encoded = np.array([EIIP_VALUES.get(base, EIIP_MEAN) for base in cleaned], dtype=np.float32)
    return encoded

def compute_base_composition(sequence: str) -> dict:
    """Compute base counts and percentages for nucleotide analysis."""
    cleaned = clean_dna_sequence(sequence)
    total = max(len(cleaned), 1)
    counts = {
        'A': cleaned.count('A'),
        'C': cleaned.count('C'),
        'G': cleaned.count('G'),
        'T': cleaned.count('T')
    }
    percentages = {base: round((count / total) * 100, 2) for base, count in counts.items()}
    gc_content = round(((counts['G'] + counts['C']) / total) * 100, 2)
    return {
        'total_length': len(cleaned),
        'counts': counts,
        'percentages': percentages,
        'gc_content': gc_content
    }
