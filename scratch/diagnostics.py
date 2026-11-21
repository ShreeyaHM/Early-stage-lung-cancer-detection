import pandas as pd, numpy as np, hashlib, collections, sys, os
sys.path.append(os.path.abspath('src'))

df = pd.read_csv('sample_data/sample_dataset.csv')

# 1. Duplicate sequences (exact)
exact_dup_counts = df['DNA_Sequence'].value_counts()
num_exact_dups = (exact_dup_counts > 1).sum()
print(f'Exact duplicate sequences: {num_exact_dups}')

# 2. Near-duplicate: hash of first 100 bp
hashes = df['DNA_Sequence'].apply(lambda s: hashlib.sha1(s[:100].encode()).hexdigest())
max_hash_count = hashes.value_counts().max()
print(f'Max count of identical first-100bp hash (potential near-dup): {max_hash_count}')

# 3. Sequence length stats
lengths = df['DNA_Sequence'].apply(len)
print('Sequence length stats:')
print(lengths.describe())

# 4. GC content per class
def gc_content(seq):
    gc = seq.count('G') + seq.count('C')
    return gc/len(seq) if len(seq)>0 else 0

df['gc'] = df['DNA_Sequence'].apply(gc_content)
print('GC content by label:')
print(df.groupby('Label')['gc'].describe())

# 5. Check for NaNs in DNA sequences
print('Any NaN DNA sequences?', df['DNA_Sequence'].isna().any())

# 6. Check label distribution
print('Label distribution:')
print(df['Label'].value_counts())

# 7. Quick feature sanity: compute one feature matrix and check for NaNs/Inf
from encoding import eiip_encode
from signal_features import extract_combined_signal_matrix
seq_example = df['DNA_Sequence'].iloc[0]
signal = eiip_encode(seq_example, target_length=500)
feature_mat = extract_combined_signal_matrix(signal, target_length=500)
print('Feature matrix shape:', feature_mat.shape)
print('Feature matrix NaNs:', np.isnan(feature_mat).any())
print('Feature matrix Infs:', np.isinf(feature_mat).any())

# 8. Check variance per channel across a subset
features = []
for seq in df['DNA_Sequence'].sample(200, random_state=42):
    sig = eiip_encode(seq, target_length=500)
    fm = extract_combined_signal_matrix(sig, target_length=500)
    features.append(fm)
features = np.stack(features)  # (N, 500, 4)
var_per_channel = features.var(axis=(0,1))
print('Variance per channel (EIIP, Envelope, InstFreq, LocalEnergy):', var_per_channel)

sys.exit(0)
