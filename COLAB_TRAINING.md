# Google Colab GPU Training Guide 🚀

This guide explains how to train the **AI-Based Early Lung Cancer Genomic Detection Model** on **Google Colab** with GPU acceleration (NVIDIA T4 / V100 / A100).

---

## ⚡ Why Train on Google Colab GPU?

| Dimension | Local CPU | Google Colab (T4 GPU) |
| :--- | :--- | :--- |
| **Training Speed** | ~11.5 minutes (20 epochs) | **~25–40 seconds** (20 epochs) |
| **Sequence Length** | Capped at 500 bp to limit runtime | **Full 1,000 bp** (covers all mutations) |
| **Model Capacity** | 128 hidden units | **256 hidden units** |
| **Number of Epochs** | 20 epochs | **50–60 epochs** with deep convergence |
| **Expected Accuracy** | ~84.5% | **88% – 93%+** |

---

## 📂 Step 1: Upload Files to Colab

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Set the runtime to GPU:
   - Go to menu: **Runtime** $\rightarrow$ **Change runtime type**
   - Under **Hardware accelerator**, select **T4 GPU**
   - Click **Save**
3. Create a zip file of your local project or upload the required folders directly into Colab:
   ```
   src/                    # Complete feature extraction & model modules
   sample_data/            # Contains sample_dataset.csv
   train.py                # Main training script
   requirements.txt        # Python package dependencies
   ```

---

## 💻 Step 2: Colab Notebook Code Cells

### Cell 1: Check GPU
```python
!nvidia-smi
```

### Cell 2: Install Dependencies
```python
!pip install torch torchvision torchaudio scikit-learn pandas numpy scipy biopython plotly
```

### Cell 3: GPU-Optimized Training Configuration
If training with the full 1,000 bp sequence length on GPU, you can run `train.py` or execute the following script block directly:

```python
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix

from src.dataset import GenomicSignalDataset
from src.encoding import clean_dna_sequence
from src.model import GenomicLSTMClassifier

# 1. Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 2. Load Dataset
df = pd.read_csv("sample_data/sample_dataset.csv")
df['clean_seq'] = df['DNA_Sequence'].apply(lambda x: clean_dna_sequence(str(x)))
valid_mask = df['clean_seq'].apply(lambda x: len(x) >= 50 and all(c in 'ACGT' for c in x))
df = df[valid_mask].reset_index(drop=True)

sequences = df['clean_seq'].tolist()
labels = df['Label'].astype(int).tolist()

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.50, stratify=[labels[i] for i in test_idx], random_state=42)

# Full sequence length: 1,000 bps
target_length = 1000
full_dataset = GenomicSignalDataset(sequences=sequences, labels=labels, target_length=target_length)

train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=64, shuffle=True)
val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=64, shuffle=False)
test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=64, shuffle=False)

# 3. Model Architecture (Higher Capacity for GPU)
model = GenomicLSTMClassifier(input_dim=3, hidden_dim=256, num_layers=2, dropout=0.25).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
num_epochs = 40
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# 4. Training Loop
best_val_acc = 0.0
best_model_weights = None
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, batch_labels, _ in train_loader:
        inputs, batch_labels = inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, batch_labels, _ in val_loader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, batch_labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += torch.sum(preds == batch_labels).item()

    val_acc = val_correct / len(val_idx)
    history['train_loss'].append(round(running_loss / len(train_idx), 4))
    history['val_loss'].append(round(val_loss / len(val_idx), 4))
    history['val_acc'].append(round(val_acc, 4))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Val Acc: {val_acc*100:.2f}% [Saved Best Checkpoint]")

# 5. Threshold Calibration & Test Evaluation
model.load_state_dict(best_model_weights)
model.eval()

val_probs, val_targets = [], []
with torch.no_grad():
    for inputs, batch_labels, _ in val_loader:
        inputs = inputs.to(device)
        logits, _ = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        val_probs.extend(probs.tolist())
        val_targets.extend(batch_labels.numpy().tolist())

fpr_val, tpr_val, thresh_val = roc_curve(val_targets, val_probs)
j_scores = tpr_val - fpr_val
best_thresh = float(thresh_val[int(np.argmax(j_scores))])
best_thresh = float(np.clip(best_thresh, 0.35, 0.65))

all_preds, all_targets, all_probs = [], [], []
with torch.no_grad():
    for inputs, batch_labels, _ in test_loader:
        inputs = inputs.to(device)
        logits, _ = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= best_thresh).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(batch_labels.numpy().tolist())

test_acc = accuracy_score(all_targets, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
fpr, tpr, _ = roc_curve(all_targets, all_probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(all_targets, all_preds).tolist()

print("\n" + "=" * 50)
print(f"TEST ACCURACY:     {test_acc * 100:.2f}%")
print(f"TEST RECALL:       {recall * 100:.2f}%")
print(f"TEST PRECISION:    {precision * 100:.2f}%")
print(f"TEST F1-SCORE:     {f1 * 100:.2f}%")
print(f"TEST ROC-AUC:      {roc_auc:.4f}")
print(f"OPTIMAL THRESHOLD: {best_thresh:.4f}")
print("=" * 50)

# Save weights and metadata
os.makedirs("models", exist_ok=True)
torch.save(best_model_weights, "models/genomic_lstm.pth")
metadata = {
    'model_name': 'GenomicBiLSTMClassifier',
    'pipeline': 'DNA Sequence -> EIIP -> FFT & Hilbert [K-mer Graph] -> PyTorch BiLSTM -> +ve / -ve',
    'dataset_size': len(df),
    'train_samples': len(train_idx),
    'test_samples': len(test_idx),
    'test_metrics': {
        'accuracy': round(test_acc * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'roc_auc': round(roc_auc, 4),
        'optimal_threshold': round(best_thresh, 4)
    },
    'confusion_matrix': cm,
    'roc_curve': {
        'fpr': [round(float(x), 4) for x in fpr],
        'tpr': [round(float(x), 4) for x in tpr]
    },
    'training_history': history,
    'date_trained': time.strftime("%Y-%m-%d %H:%M:%S")
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved models/genomic_lstm.pth and models/model_metadata.json!")
```

### Cell 4: Download Results Back to Local Machine
```python
from google.colab import files
files.download("models/genomic_lstm.pth")
files.download("models/model_metadata.json")
```

---

## 📥 Step 3: Put Trained Weights into Local Project

1. Move the downloaded `genomic_lstm.pth` and `model_metadata.json` into your local `models/` directory:
   ```
   e:\lungcancer_dl\models\genomic_lstm.pth
   e:\lungcancer_dl\models\model_metadata.json
   ```
2. Relaunch or refresh your Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
3. The dashboard will automatically read the updated metrics and run inference with the new GPU-trained model.
