"""
Training Script: Genomic BiLSTM Classifier for Early Lung Cancer Detection
Trains the PyTorch LSTM model on EIIP + FFT + Hilbert features of lung cancer genomic signals.
Loads training data from sample_data/sample_dataset.csv (columns: DNA_Sequence, Label).
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix

from src.dataset import GenomicSignalDataset
from src.encoding import clean_dna_sequence
from src.model import GenomicLSTMClassifier

def train():
    print("STARTING TRAINING", flush=True)
    print("=" * 65)
    print("[PROJECT] AI-Based Early Lung Cancer Detection from Genomic Signals")
    print("[MODEL] Training PyTorch BiLSTM (EIIP + FFT + Hilbert) Model")
    print("=" * 65)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # 1. Load dataset from CSV
    csv_path = "sample_data/sample_dataset.csv"
    print(f"\n[Step 1/5] Loading genomic dataset from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset file not found at {csv_path}")
        print("Please place your dataset CSV with columns 'DNA_Sequence' and 'Label' in sample_data/")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Dataset size: {len(df)}", flush=True)
    print("DATASET LOADED", flush=True)
    print(f"Dataset loaded: {len(df)} sequences")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    
    # Clean DNA sequences - keep only valid ACGT characters
    df['clean_seq'] = df['DNA_Sequence'].apply(lambda x: clean_dna_sequence(str(x)))
    
    # Filter out invalid sequences (must have only ACGT and be at least 50bp)
    valid_mask = df['clean_seq'].apply(lambda x: len(x) >= 50 and all(c in 'ACGT' for c in x))
    df = df[valid_mask].reset_index(drop=True)
    print(f"Valid sequences after cleaning: {len(df)}")
    
    sequences = df['clean_seq'].tolist()
    labels = df['Label'].astype(int).tolist()
    
    # 2. Stratified Split (70% train, 15% val, 15% test)
    print("\n[Step 2/5] Preparing feature tensors and train/val/test splits...", flush=True)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.50, stratify=[labels[i] for i in test_idx], random_state=42)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}", flush=True)
    print("FEATURE EXTRACTION COMPLETE", flush=True)
    
    # Create full dataset (binary classification only - no subtypes)
    target_length = 500
    full_dataset = GenomicSignalDataset(
        sequences=sequences,
        labels=labels,
        target_length=target_length
    )
    # Feature shape diagnostics
    print(f"Feature shape: {full_dataset[0][0].shape}", flush=True)

    # Diagnostic prints to verify feature matrix dimensions and integrity
    sample_matrix = full_dataset[0][0]  # first element returns (features, label, idx?)
    print("Feature matrix shape:", sample_matrix.shape)
    print("Feature channels:", sample_matrix.shape[-1])
    # Check for NaNs or Infs
    if np.isnan(sample_matrix).any():
        print("[WARNING] Feature matrix contains NaN values.")
    if np.isinf(sample_matrix).any():
        print("[WARNING] Feature matrix contains Inf values.")
    # Verify variation per channel
    variations = sample_matrix.var(axis=0)
    print("Channel variances:", variations)
    if (variations == 0).any():
        print("[WARNING] One or more channels have zero variance (constant).")

    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=32, shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=32, shuffle=False, drop_last=False)
    
    # 3. Initialize Model, Loss, Optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Using device: {device}")
    
    model = GenomicLSTMClassifier(input_dim=3, hidden_dim=128, num_layers=2, dropout=0.20).to(device)
    print("MODEL INITIALIZED", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    # 4. Training Loop
    num_epochs = 20
    print(f"\n[Step 3/5] Commencing binary model training for {num_epochs} epochs...", flush=True)
    print("STARTING EPOCH LOOP", flush=True)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_weights = None
    patience_counter = 0
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, batch_labels, _ in train_loader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_idx)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, batch_labels, _ in val_loader:
                inputs = inputs.to(device)
                batch_labels = batch_labels.to(device)
                
                logits, _ = model(inputs)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(logits, dim=1)
                val_correct += torch.sum(preds == batch_labels).item()
                
        epoch_val_loss = val_loss / len(val_idx)
        epoch_val_acc = val_correct / len(val_idx)
        scheduler.step(epoch_val_acc)
        
        history['train_loss'].append(round(epoch_train_loss, 4))
        history['val_loss'].append(round(epoch_val_loss, 4))
        history['val_acc'].append(round(epoch_val_acc, 4))
        
        saved_msg = ""
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(best_model_weights, "models/genomic_lstm.pth")
            saved_msg = f" [Saved Best Checkpoint: {best_val_acc*100:.1f}%]"
            patience_counter = 0
        else:
            patience_counter += 1
            
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%{saved_msg}", flush=True)
        
        # Early stopping if target accuracy reached or plateaued
        if epoch_val_acc >= 0.92 and (epoch + 1) >= 8:
            print(f"Target accuracy reached (Val Acc: {epoch_val_acc*100:.1f}% >= 92%). Early stopping triggered!", flush=True)
            break
        if patience_counter >= 5 and (epoch + 1) >= 10:
            print("Early stopping triggered due to validation plateau.", flush=True)
            break
            
    train_duration = round(time.time() - start_time, 2)
    print(f"\nTraining completed in {train_duration}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # Load best weights
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, "models/genomic_lstm.pth")
    print("Saved best model weights to: models/genomic_lstm.pth")
    
    # 5. Evaluation on Unseen Test Set
    print("\n[Step 4/5] Evaluating on unseen test set...")
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, batch_labels, _ in test_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_labels.numpy().tolist())
            
    test_acc = accuracy_score(all_targets, all_preds) 
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    # Optimal Youden's J statistic threshold
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_thresh = float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.50
    optimal_thresh = float(np.clip(optimal_thresh, 0.40, 0.85))
    
    print(f"Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"Test Precision:      {precision * 100:.2f}%")
    print(f"Test Recall:         {recall * 100:.2f}%")
    print(f"Test F1-Score:       {f1 * 100:.2f}%")
    print(f"Test ROC-AUC:        {roc_auc:.4f}")
    print(f"Optimal Threshold:   {optimal_thresh:.4f}")
    
    # 6. Save Model Metadata & Analytics
    print("\n[Step 5/5] Exporting metadata and evaluation records...")
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
            'optimal_threshold': round(optimal_thresh, 4)
        },
        'confusion_matrix': cm,
        'roc_curve': {
            'fpr': [round(float(x), 4) for x in fpr],
            'tpr': [round(float(x), 4) for x in tpr]
        },
        'training_history': history,
        'train_duration_sec': train_duration,
        'date_trained': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to: models/model_metadata.json")
    print("\n[SUCCESS] All training and validation procedures completed successfully!")

if __name__ == '__main__':
    train()
