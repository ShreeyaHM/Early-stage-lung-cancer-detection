"""
Genomic Dataset Management & Synthetic Curated Sequence Generator
Constructs biologically authentic training & test datasets for Lung Cancer Detection
using known somatic mutation hotspots (TP53, EGFR, KRAS) and healthy wild-type alleles.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

from src.encoding import eiip_encode, clean_dna_sequence
from src.signal_features import extract_combined_signal_matrix

# Authentic human wild-type reference DNA coding sequences (excerpts)
REFERENCE_SEQUENCES = {
    'TP53_WT': (
        "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTT"
        "CTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGA"
        "AGCTCCCAGAATGCCAGAGGCTGCTCCCCGCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCC"
        "CTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACT"
        "TGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGC"
        "ACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGC"
        "GATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTG"
        "GTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAAC"
        "CGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCT"
        "GGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCC"
    ),
    'EGFR_WT': (
        "ATGCGACCCTCCGGGACGGCCGGGGCAGCGCTCCTGGCGCTGCTGGCTGCGCTCTGCCCGGCGAGTCGGGCTCTGGAGGAAAAGAAAGTTTGC"
        "CAAGGCACGAGTAACAAGCTCACGCAGTTGGGCACTTTTGAAGATCATTTTCTCAGCCTCCAGAGGATGTTCAATAACTGTGAGGTGGTCCTT"
        "GGGAATTTGGAAATTACCTATGTGCAGAGGAATTATGATCTTTCCTTCTTTAAGACCATCCAGGAGGTGGCTGGTTATGTCCTCATTGCCCTC"
        "AACACAGTGGAGCGAATTCCTTTGGAAAACCTGCAGATCATCAGAGGAAATATGTACTACGAAAATTCCTATGCCTTAGCAGTCTTATCTAAC"
        "TATGATGCAAATAAAACCGGACTGAAGGAGCTGCCCATGAGAAATTTACAGGAAATCCTGCATGGCGCCGTGCGGTTCAGCAACAACCCTGCC"
        "CTGTGCAACGTGGAGAGCATCCAGTGGCGGGACATAGTCAGCAGTGACTTTCTCAGCAACATGTCGATGGACTTCCAGAACCACCTGGGCAGC"
        "TGCCAAAAGTGTGATCCAAGCTGTCCCAATGGGAGCTGCTGGGGTGCAGGAGAGGAGAACTGCCAGAAACTGACCAAAATCATCTGTGCCCAG"
        "CAGTGCTCCGGGCGCTGCCGTGGCAAGTCCCCCAGTGACTGCTGCCACAACCAGTGTGCTGCAGGCTGCACAGGCCCCCGGGAGAGCGACTGC"
        "CTGGTCTGCCGCAAATTCCGAGACGAAGCCACGTGCAAGGACACCTGCCCCCCACTCATGCTCTACAACCCCACCACGTACCAGATGGATGTG"
    ),
    'KRAS_WT': (
        "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAA"
        "TATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTAATTGATGGAGAAACCTGTCTCTTGGATATTCTCGACACAGCAGGTCAAGAG"
        "GAGTACAGTGCAATGAGGGACCAGTACATGAGGACTGGGGAGGGCTTTCTTTGTGTATTTGCCATAAATAATACTAAATCATTTGAAGATATT"
        "CACCATTATAGAGAACAAATTAAAAGAGTTAAGGACTCTGAAGATGTACCTATGGTCCTAGTAGGAAATAAATGTGATTTGCCTTCTAGAACA"
        "GTAGACACAAAACAGGCTCAGGACTTAGCAAGAAGTTATGGAATTCCTTTTATTGAAACATCAGCAAAGACAAGACAGGGTGTTGATGATGCA"
        "TTCTACACATTAGTTCGAGAAATTCGAAAACATAAAGAAAAGATGAGCAAAGATGGTAAAAAGAAGAAAAAGAAGTCAAAGACAAAGTGTGTA"
    ),
    'NORMAL_PULM_WT': (
        "ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTC"
        "GCCATCAATGACCCCTTCATTGACCTCAACTACATGGTTTACATGTTCCAATATGATTCCACCCATGGCAAATTCCATGGCACCGTCAAGGCT"
        "GAGAACGGGAAGCTTGTCATCAATGGAAATCCCATCACCATCTTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAG"
        "TACGTCGTGGAGTCCACTGGCGTCTTCACCACCATGGAGAAGGCTGGGGCTCATTTGCAGGGGGGAGCCAAAAGGGTCATCATCTCTGCCCCCT"
        "CTGCTGATGCCCCCATGTTCGTCATGGGTGTGAACCATGAGAAGTATGACAACAGCCTCAAGATCATCAGCAATGCCTCCTGCACCACCAACTG"
    )
}

# Known Lung Cancer Hotspot Mutations
MUTATION_SIGNATURES = {
    'TP53': [
        {'name': 'R175H (CGC->CAC)', 'pos_range': (200, 260), 'motif': 'CGC', 'mutant': 'CAC'},
        {'name': 'R248W (CGG->TGG)', 'pos_range': (350, 420), 'motif': 'CGG', 'mutant': 'TGG'},
        {'name': 'R273H (CGT->CAT)', 'pos_range': (420, 500), 'motif': 'CGT', 'mutant': 'CAT'},
        {'name': 'G245S (GGC->AGC)', 'pos_range': (340, 410), 'motif': 'GGC', 'mutant': 'AGC'},
    ],
    'EGFR': [
        {'name': 'Exon 21 L858R (CTG->CGG)', 'pos_range': (300, 450), 'motif': 'CTG', 'mutant': 'CGG'},
        {'name': 'Exon 20 T790M (ACG->ATG)', 'pos_range': (250, 380), 'motif': 'ACG', 'mutant': 'ATG'},
        {'name': 'Exon 19 delE746_A750', 'pos_range': (200, 320), 'motif': 'AATTAAGAGAAGC', 'mutant': 'AAGC'},
    ],
    'KRAS': [
        {'name': 'Codon 12 G12C (GGT->TGT)', 'pos_range': (30, 45), 'motif': 'GGT', 'mutant': 'TGT'},
        {'name': 'Codon 12 G12D (GGT->GAT)', 'pos_range': (30, 45), 'motif': 'GGT', 'mutant': 'GAT'},
        {'name': 'Codon 12 G12V (GGT->GTT)', 'pos_range': (30, 45), 'motif': 'GGT', 'mutant': 'GTT'},
    ]
}

def inject_somatic_mutation(sequence: str, gene: str) -> Tuple[str, str]:
    """
    Inject a known clinical lung cancer driver mutation into the wild-type reference sequence,
    along with realistic somatic background point mutations (COSMIC Signature 4).
    """
    seq_list = list(sequence)
    mutations = MUTATION_SIGNATURES.get(gene, [])
    if not mutations:
        return sequence, "Wild-Type"
        
    chosen = random.choice(mutations)
    sub = chosen['mutant']
    motif = chosen['motif']
    
    # Try finding the motif in the sequence
    sub_seq = sequence
    idx = sub_seq.find(motif)
    if idx != -1:
        # Replace motif with mutant
        seq_list[idx:idx+len(motif)] = list(sub)
    else:
        # Fallback to random position in pos_range
        p_start, p_end = chosen['pos_range']
        pos = min(random.randint(p_start, min(p_end, len(seq_list) - len(sub))), len(seq_list) - len(sub))
        seq_list[pos:pos+len(sub)] = list(sub)
        
    # Inject somatic signature point variations (C>A transversions and hotspot alterations)
    num_snvs = random.randint(3, 6)
    for _ in range(num_snvs):
        p = random.randint(0, len(seq_list) - 1)
        if seq_list[p] == 'C':
            seq_list[p] = 'A' # Signature 4: tobacco smoking C>A transversion
        elif seq_list[p] == 'G':
            seq_list[p] = 'T'
        else:
            seq_list[p] = random.choice(['C', 'G'])
            
    mutated_seq = "".join(seq_list)
    return mutated_seq, chosen['name']

def generate_genomic_dataset(num_samples: int = 400, seq_len: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced dataset of normal wild-type and cancerous lung sequences.
    
    Classes:
      - Normal Control: label = 0, subtype = 0
      - TP53 Alteration: label = 1, subtype = 1
      - EGFR Mutation: label = 1, subtype = 2
      - KRAS Mutation: label = 1, subtype = 3
    """
    random.seed(seed)
    np.random.seed(seed)
    
    records = []
    
    # Exactly balanced generation: 200 Cancer Positive (+ve) vs 200 Normal Negative (-ve)
    gene_configs = [
        ('TP53', 'TP53_WT', 1),
        ('EGFR', 'EGFR_WT', 2),
        ('KRAS', 'KRAS_WT', 3),
        ('PULM', 'NORMAL_PULM_WT', 1)
    ]
    
    samples_per_gene = num_samples // len(gene_configs) # 100 per gene
    
    for gene_name, ref_key, subtype_id in gene_configs:
        base_seq = REFERENCE_SEQUENCES[ref_key]
        reps = (seq_len // len(base_seq)) + 2
        extended_seq = base_seq * reps
        
        for i in range(samples_per_gene):
            max_start = max(0, len(base_seq) - seq_len)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            seq_window = extended_seq[start_idx:start_idx+seq_len]
            
            # 50% Cancer Positive vs 50% Normal Negative
            is_cancer_sample = (i % 2 == 1)
            
            if is_cancer_sample:
                target_gene = 'TP53' if gene_name == 'PULM' else gene_name
                mut_seq, mut_info = inject_somatic_mutation(seq_window, target_gene)
                records.append({
                    'sequence_id': f"CANCER_{gene_name}_{i+1:04d}",
                    'sequence': mut_seq,
                    'label': 1, # Cancer Positive (+ve)
                    'subtype': subtype_id,
                    'gene': f"Cancer ({gene_name})",
                    'mutation_info': f"Somatic Alteration: {mut_info}"
                })
            else:
                # Benign wild-type control
                seq_chars = list(seq_window)
                if random.random() < 0.15:
                    p = random.randint(0, len(seq_chars) - 1)
                    seq_chars[p] = random.choice(['A', 'C', 'G', 'T'])
                wt_seq = "".join(seq_chars)
                records.append({
                    'sequence_id': f"NORMAL_{gene_name}_{i+1:04d}",
                    'sequence': wt_seq,
                    'label': 0, # Normal Negative (-ve)
                    'subtype': 0,
                    'gene': f"Normal ({gene_name} WT)",
                    'mutation_info': 'Benign Wild-Type Control'
                })
                
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

class GenomicSignalDataset(Dataset):
    """
    PyTorch Dataset that maps DNA sequences to multi-channel feature matrices
    using EIIP numerical encoding, Hilbert transform, and FFT local energy.
    """
    def __init__(self, sequences: List[str], labels: List[int], subtypes: Optional[List[int]] = None, target_length: int = 500):
        self.sequences = sequences
        self.labels = labels
        self.subtypes = subtypes if subtypes is not None else [0] * len(labels)
        self.target_length = target_length
        
        # Precompute feature matrices for fast training
        self.features = []
        for seq in sequences:
            eiip_sig = eiip_encode(seq, target_length=self.target_length)
            matrix = extract_combined_signal_matrix(eiip_sig, target_length=self.target_length)
            self.features.append(matrix)
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subtype = torch.tensor(self.subtypes[idx], dtype=torch.long)
        return feat, label, subtype
