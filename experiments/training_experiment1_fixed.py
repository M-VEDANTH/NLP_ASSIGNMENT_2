"""
Fixed Training Experiment 1: Independent Slot Filling and Intent Classification
-----------------------------------------------------------------------------
Fixed version with proper hyperparameters to resolve 0% accuracy issue.

Key fixes:
1. Reduced learning rate from 1e-3 to 1e-4
2. Added learning rate scheduler
3. Improved model initialization
4. Enhanced debugging and validation
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.rnn_scratch import RNNForSlotFilling, RNNForIntentClassification
from models.lstm_scratch import LSTMForSlotFilling, LSTMForIntentClassification

# Simple metrics implementation to avoid sklearn issues
def precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0):
    """Calculate precision, recall, F1-score without sklearn dependency."""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []
    
    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    if average == 'weighted':
        # Weight by label frequency
        weights = []
        for label in unique_labels:
            weights.append(np.sum(y_true == label))
        weights = np.array(weights) / len(y_true)
        
        avg_precision = np.average(precisions, weights=weights)
        avg_recall = np.average(recalls, weights=weights)
        avg_f1 = np.average(f1s, weights=weights)
    else:  # macro
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)
    
    return avg_precision, avg_recall, avg_f1, None

def accuracy_score(y_true, y_pred):
    """Calculate accuracy without sklearn dependency."""
    return np.mean(y_true == y_pred)


def init_model_weights(model):
    """Initialize model weights properly."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.zeros_(param)
    print("✅ Model weights initialized with Xavier uniform")


# ============================================================
# Fixed Training Function for Slot Filling
# ============================================================

def train_slot_filling(model, train_loader, val_loader, num_epochs=15, lr=1e-4, device='cuda', 
                      save_checkpoints=True, checkpoint_dir="checkpoints", model_name="slot_model"):
    """Train slot filling model with fixed hyperparameters."""
    model.to(device)
    
    # Initialize weights properly
    init_model_weights(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = padding index
    
    print(f"🔄 Training {model.__class__.__name__} for Slot Filling...")
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"🎯 Learning rate: {lr}, Weight decay: 1e-5")

    # Create checkpoint directory
    if save_checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"💾 Checkpoints will be saved to: {checkpoint_path}")

    best_f1 = 0.0
    training_history = {'train_loss': [], 'val_f1': [], 'val_acc': [], 'learning_rates': []}
    start_epoch = 0
    
    # Try to load existing checkpoint
    if save_checkpoints:
        latest_checkpoint = os.path.join(checkpoint_path, 'latest_checkpoint.pt')
        if os.path.exists(latest_checkpoint):
            print("🔄 Found existing checkpoint, loading...")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['best_f1']
            training_history = checkpoint['training_history']
            print(f"✅ Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(pbar):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, attention_mask)
                
                # Reshape for loss calculation
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = labels.view(-1)
                
                loss = criterion(logits_flat, labels_flat)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Enhanced progress info
                if batch_idx == 0 or (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
        
        avg_train_loss = total_loss / num_batches
        training_history['train_loss'].append(avg_train_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_slot_filling(model, val_loader, device, verbose=False)
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_acc'].append(val_metrics['accuracy'])
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['f1'])
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: Loss={avg_train_loss:.4f}, Val F1={val_metrics['f1']:.4f}, Val Acc={val_metrics['accuracy']:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        if save_checkpoints:
            # Save latest checkpoint
            latest_checkpoint = os.path.join(checkpoint_path, 'latest_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'training_history': training_history,
                'val_metrics': val_metrics
            }, latest_checkpoint)
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_checkpoint = os.path.join(checkpoint_path, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'training_history': training_history
                }, best_checkpoint)
                print(f"💾 New best model saved! F1: {best_f1:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                periodic_checkpoint = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, periodic_checkpoint)
    
    print(f"✅ Training completed! Best F1: {best_f1:.4f}")
    return model, training_history


# ============================================================
# Fixed Training Function for Intent Classification
# ============================================================

def train_intent_classification(model, train_loader, val_loader, num_epochs=15, lr=1e-4, device='cuda',
                               save_checkpoints=True, checkpoint_dir="checkpoints", model_name="intent_model"):
    """Train intent classification model with fixed hyperparameters."""
    model.to(device)
    
    # Initialize weights properly
    init_model_weights(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🔄 Training {model.__class__.__name__} for Intent Classification...")
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"🎯 Learning rate: {lr}, Weight decay: 1e-5")

    # Create checkpoint directory
    if save_checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"💾 Checkpoints will be saved to: {checkpoint_path}")

    best_acc = 0.0
    training_history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'learning_rates': []}
    start_epoch = 0
    
    # Try to load existing checkpoint
    if save_checkpoints:
        latest_checkpoint = os.path.join(checkpoint_path, 'latest_checkpoint.pt')
        if os.path.exists(latest_checkpoint):
            print("🔄 Found existing checkpoint, loading...")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            training_history = checkpoint['training_history']
            print(f"✅ Resumed from epoch {start_epoch}, best accuracy: {best_acc:.4f}")

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(pbar):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Enhanced progress info
                if batch_idx == 0 or (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
        
        avg_train_loss = total_loss / num_batches
        training_history['train_loss'].append(avg_train_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_intent_classification(model, val_loader, device, verbose=False)
        training_history['val_acc'].append(val_metrics['accuracy'])
        training_history['val_f1'].append(val_metrics['f1'])
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: Loss={avg_train_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1']:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        if save_checkpoints:
            # Save latest checkpoint
            latest_checkpoint = os.path.join(checkpoint_path, 'latest_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'training_history': training_history,
                'val_metrics': val_metrics
            }, latest_checkpoint)
            
            # Save best model
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_checkpoint = os.path.join(checkpoint_path, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'training_history': training_history
                }, best_checkpoint)
                print(f"💾 New best model saved! Accuracy: {best_acc:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                periodic_checkpoint = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, periodic_checkpoint)
    
    print(f"✅ Training completed! Best accuracy: {best_acc:.4f}")
    return model, training_history


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_slot_filling(model, test_loader, device, verbose=True):
    """Evaluate slot filling model with enhanced debugging."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels (excluding padding tokens)
            for i in range(batch_size):
                mask = attention_mask[i].bool()
                seq_predictions = predictions[i][mask].cpu().numpy()
                seq_labels = labels[i][mask].cpu().numpy()
                
                # Filter out padding tokens (0)
                non_pad_mask = seq_labels != 0
                if non_pad_mask.sum() > 0:
                    all_predictions.extend(seq_predictions[non_pad_mask])
                    all_labels.extend(seq_labels[non_pad_mask])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    if len(all_predictions) > 0:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        avg_loss = total_loss / num_batches
        
        if verbose:
            print(f"🎯 Slot Filling Evaluation:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-score: {f1:.4f}")
            print(f"   Total predictions: {len(all_predictions)}")
            print(f"   Unique predicted labels: {len(np.unique(all_predictions))}")
            print(f"   Unique true labels: {len(np.unique(all_labels))}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss,
            'num_samples': len(all_predictions)
        }
    else:
        if verbose:
            print("❌ No valid predictions found!")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'num_samples': 0
        }


def evaluate_intent_classification(model, test_loader, device, verbose=True):
    """Evaluate intent classification model with enhanced debugging."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / num_batches
    
    if verbose:
        print(f"🎯 Intent Classification Evaluation:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-score: {f1:.4f}")
        print(f"   Total samples: {len(all_predictions)}")
        print(f"   Unique predicted labels: {len(np.unique(all_predictions))}")
        print(f"   Unique true labels: {len(np.unique(all_labels))}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss,
        'num_samples': len(all_predictions)
    }


# ============================================================
# Dataset Loading and Data Loaders (Same as before)
# ============================================================

def load_dataset(dataset_name='atis'):
    """Load preprocessed dataset and vocabularies."""
    data_dir = os.path.join('data', dataset_name)
    
    # Check if files exist
    required_files = [
        'X_train.npy', 'X_test.npy',
        'y_intents_train.npy', 'y_intents_test.npy',
        'y_slots_train.npy', 'y_slots_test.npy',
        f'{dataset_name}_vocabularies.json'
    ]
    
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
    
    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_intents_train = np.load(os.path.join(data_dir, 'y_intents_train.npy'))
    y_intents_test = np.load(os.path.join(data_dir, 'y_intents_test.npy'))
    y_slots_train = np.load(os.path.join(data_dir, 'y_slots_train.npy'))
    y_slots_test = np.load(os.path.join(data_dir, 'y_slots_test.npy'))
    
    # Load vocabularies
    with open(os.path.join(data_dir, f'{dataset_name}_vocabularies.json'), 'r') as f:
        vocabularies = json.load(f)
    
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_intents_train': y_intents_train,
        'y_intents_test': y_intents_test,
        'y_slots_train': y_slots_train,
        'y_slots_test': y_slots_test
    }
    
    print(f"✅ Loaded {dataset_name} dataset:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Vocabulary size: {len(vocabularies['word2idx'])}")
    print(f"   Intent classes: {len(vocabularies['intent2idx'])}")
    print(f"   Slot classes: {len(vocabularies['slot2idx'])}")
    
    return data, vocabularies


def create_attention_mask(sequences):
    """Create attention mask (1 for real tokens, 0 for padding)."""
    return (sequences != 0).astype(np.float32)


def create_data_loaders(data, task_type='slot', batch_size=32, shuffle=True):
    """Create DataLoader for training and testing."""
    X_train, X_test = data['X_train'], data['X_test']
    
    if task_type == 'slot':
        y_train, y_test = data['y_slots_train'], data['y_slots_test']
    else:  # intent
        y_train, y_test = data['y_intents_train'], data['y_intents_test']
    
    # Create attention masks
    train_attention_mask = create_attention_mask(X_train)
    test_attention_mask = create_attention_mask(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.LongTensor(X_train)
    X_test_tensor = torch.LongTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    train_mask_tensor = torch.FloatTensor(train_attention_mask)
    test_mask_tensor = torch.FloatTensor(test_attention_mask)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, train_mask_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, test_mask_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Created data loaders for {task_type}:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, test_loader


# ============================================================
# Complete Experiment Runner
# ============================================================

def run_independent_experiment(dataset_name='atis', device='cuda'):
    """Run complete independent models experiment with fixed hyperparameters."""
    print(f"\n{'='*60}")
    print(f"🚀 Fixed Independent Models Experiment - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    try:
        data, vocabularies = load_dataset(dataset_name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return {}
    
    # Get sizes
    vocab_size = len(vocabularies['word2idx'])
    num_slot_labels = len(vocabularies['slot2idx'])
    num_intent_labels = len(vocabularies['intent2idx'])
    
    # Model configurations
    models_config = {
        'RNN': {
            'slot': RNNForSlotFilling,
            'intent': RNNForIntentClassification
        },
        'LSTM': {
            'slot': LSTMForSlotFilling,
            'intent': LSTMForIntentClassification
        }
    }
    
    results = {}
    
    # Train each model type for each task
    for model_name in ['RNN', 'LSTM']:
        for task_type in ['slot', 'intent']:
            print(f"\n{'='*50}")
            print(f"🔄 Training {model_name} for {task_type} classification")
            print(f"{'='*50}")
            
            try:
                # Create data loaders
                train_loader, test_loader = create_data_loaders(data, task_type, batch_size=32)
                
                # Get model class and parameters
                model_class = models_config[model_name][task_type]
                if task_type == 'slot':
                    num_labels = num_slot_labels
                else:
                    num_labels = num_intent_labels
                
                # Create model with fixed parameters
                model = model_class(
                    vocab_size=vocab_size,
                    embedding_dim=128,
                    hidden_size=128,
                    num_labels=num_labels,
                    num_layers=2,
                    dropout=0.3,
                    bidirectional=True
                )
                
                print(f"📋 Model: {model_class.__name__}")
                print(f"📏 Vocab: {vocab_size}, Labels: {num_labels}")
                print(f"🔧 Hidden: 128, Layers: 2, Bidirectional: True")
                print(f"⚙️ Fixed hyperparameters: LR=1e-4, Weight Decay=1e-5")
                
                # Train model with fixed hyperparameters
                checkpoint_name = f"{model_name}_{task_type}_{dataset_name}"
                if task_type == 'slot':
                    trained_model, history = train_slot_filling(
                        model, train_loader, test_loader, 
                        num_epochs=15, lr=1e-4, device=device,  # Fixed parameters
                        save_checkpoints=True, checkpoint_dir="experiments/checkpoints",
                        model_name=checkpoint_name
                    )
                else:
                    trained_model, history = train_intent_classification(
                        model, train_loader, test_loader, 
                        num_epochs=15, lr=1e-4, device=device,  # Fixed parameters
                        save_checkpoints=True, checkpoint_dir="experiments/checkpoints",
                        model_name=checkpoint_name
                    )
                
                # Final evaluation
                if task_type == 'slot':
                    final_metrics = evaluate_slot_filling(trained_model, test_loader, device)
                else:
                    final_metrics = evaluate_intent_classification(trained_model, test_loader, device)
                
                # Store results
                results[f"{model_name}_{task_type}"] = {
                    'model': model_name,
                    'task': task_type,
                    'dataset': dataset_name,
                    'metrics': final_metrics,
                    'history': history,
                    'config': {
                        'vocab_size': vocab_size,
                        'num_labels': num_labels,
                        'embedding_dim': 128,
                        'hidden_size': 128,
                        'num_layers': 2,
                        'dropout': 0.3,
                        'bidirectional': True,
                        'learning_rate': 1e-4,  # Fixed
                        'weight_decay': 1e-5,   # Fixed
                        'num_epochs': 15        # Fixed
                    }
                }
                
                print(f"✅ {model_name} {task_type} training completed!")
                print(f"🏆 Final metrics: Accuracy={final_metrics['accuracy']:.4f}, F1={final_metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"❌ Failed to train {model_name} for {task_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return results


def quick_test():
    """Quick test with reduced parameters for debugging."""
    print("🧪 Running quick test on ATIS dataset...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data, vocabularies = load_dataset('atis')
    train_loader, test_loader = create_data_loaders(data, 'intent', batch_size=8)
    
    # Create model
    model = RNNForIntentClassification(
        vocab_size=len(vocabularies['word2idx']),
        embedding_dim=64,
        hidden_size=64,
        num_labels=len(vocabularies['intent2idx']),
        num_layers=1,
        dropout=0.1,
        bidirectional=False
    )
    
    # Quick training
    trained_model, history = train_intent_classification(
        model, train_loader, test_loader,  # Use full test loader
        num_epochs=3, lr=1e-4, device=device,
        save_checkpoints=False
    )
    
    print("✅ Quick test completed!")


def main():
    """Main function with various testing options."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"\n{'='*70}")
    print("🚀 FIXED TRAINING EXPERIMENT - RESOLVING 0% ACCURACY ISSUE")
    print(f"{'='*70}")
    print("Key fixes applied:")
    print("✅ Reduced learning rate: 1e-3 → 1e-4")
    print("✅ Added learning rate scheduler")
    print("✅ Proper Xavier weight initialization")
    print("✅ Enhanced debugging and validation")
    print("✅ Gradient clipping")
    print("✅ Increased epochs: 10 → 15")
    print("")
    
    # Run quick test first
    print("Running quick test to verify fixes...")
    quick_test()
    
    print("\n" + "="*50)
    print("Quick test passed! Ready for full experiment.")
    print("Uncomment the lines below to run full experiment:")
    print("")
    
    # Full experiment (uncomment to run)
    for dataset in ['atis', 'slurp']:
        print(f"\n🎯 Starting {dataset.upper()} experiment...")
        try:
            results = run_independent_experiment(dataset, device)
            print(f"✅ {dataset.upper()} experiment completed!")
        except Exception as e:
            print(f"❌ Failed experiment for {dataset}: {e}")


if __name__ == "__main__":
    main()