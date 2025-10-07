"""
Experiment 3: Intent → Slot Pipeline
-----------------------------------
Pipeline experiment where intent classification predictions are used as features for slot filling.

Architecture:
1. Train intent classification model first (from scratch or load checkpoint)
2. Use intent predictions as additional features for slot filling
3. Compare with independent models to measure pipeline benefit

The pipeline works as follows:
- Input text → Intent Model → Intent prediction
- [Input text embeddings + Intent prediction] → Slot Model → Slot predictions
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
from experiments.training_experiment1_fixed import (
    precision_recall_fscore_support, accuracy_score, init_model_weights,
    load_dataset, create_attention_mask, evaluate_slot_filling, evaluate_intent_classification
)


# ============================================================
# Pipeline-Specific Models
# ============================================================

class IntentToSlotPipelineModel(nn.Module):
    """
    Slot filling model that uses intent predictions as additional features.
    
    Architecture:
    - Word embeddings + Intent prediction embeddings
    - Combined features fed to RNN/LSTM
    - Final slot classification layer (sequence labeling)
    """
    
    def __init__(self, vocab_size, num_intent_labels, num_slot_labels, 
                 embedding_dim=128, intent_embedding_dim=32, hidden_size=128, 
                 num_layers=2, dropout=0.3, bidirectional=True, model_type='LSTM'):
        super(IntentToSlotPipelineModel, self).__init__()
        
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Intent prediction embeddings (to convert intent predictions to dense features)
        # We'll broadcast the intent prediction to all sequence positions
        self.intent_embedding = nn.Embedding(num_intent_labels, intent_embedding_dim)
        
        # Combined input size
        combined_input_size = embedding_dim + intent_embedding_dim
        
        # RNN/LSTM layer
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:  # RNN
            self.rnn = nn.RNN(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        # Output size adjustment for bidirectional
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Dropout and slot classification layers
        self.dropout = nn.Dropout(dropout)
        self.slot_classifier = nn.Linear(rnn_output_size, num_slot_labels)
        
        print(f"✅ Created {model_type} Intent→Slot Pipeline Model:")
        print(f"   Word embedding: {vocab_size} → {embedding_dim}")
        print(f"   Intent embedding: {num_intent_labels} → {intent_embedding_dim}")
        print(f"   Combined input: {combined_input_size}")
        print(f"   RNN output: {rnn_output_size}")
        print(f"   Slot classes: {num_slot_labels}")
        
    def forward(self, input_ids, intent_predictions, attention_mask=None):
        """
        Forward pass with both word tokens and intent predictions.
        
        Args:
            input_ids: [batch_size, seq_len] - word token ids
            intent_predictions: [batch_size] - predicted intent labels (single label per sequence)
            attention_mask: [batch_size, seq_len] - attention mask
            
        Returns:
            logits: [batch_size, seq_len, num_slot_labels] - slot classification logits
        """
        batch_size, seq_len = input_ids.size()
        
        # Get word embeddings
        word_embeddings = self.word_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Get intent embeddings and broadcast to all sequence positions
        intent_embeddings = self.intent_embedding(intent_predictions)  # [batch_size, intent_embedding_dim]
        intent_embeddings = intent_embeddings.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, intent_embedding_dim]
        
        # Combine word and intent embeddings
        combined_embeddings = torch.cat([word_embeddings, intent_embeddings], dim=-1)
        # [batch_size, seq_len, embedding_dim + intent_embedding_dim]
        
        # Apply attention mask to embeddings if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(combined_embeddings)
            combined_embeddings = combined_embeddings * mask_expanded
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(combined_embeddings)
        # rnn_output: [batch_size, seq_len, hidden_size * num_directions]
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Slot classification for each token
        slot_logits = self.slot_classifier(rnn_output)
        # slot_logits: [batch_size, seq_len, num_slot_labels]
        
        return slot_logits


# ============================================================
# Pipeline Training Functions
# ============================================================

def extract_intent_predictions(intent_model, data_loader, device):
    """
    Extract intent predictions from a trained intent model.
    
    Returns:
        predictions: List of numpy arrays with intent predictions for each sample
        input_ids: List of numpy arrays with input token ids for each sample
        attention_masks: List of numpy arrays with attention masks for each sample
        true_slots: List of true slot labels
    """
    intent_model.eval()
    all_predictions = []
    all_input_ids = []
    all_attention_masks = []
    all_true_slots = []
    
    print("🔄 Extracting intent predictions for pipeline...")
    
    with torch.no_grad():
        for input_ids, attention_mask, slot_labels in tqdm(data_loader, desc="Extracting predictions"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get intent predictions
            intent_logits = intent_model(input_ids, attention_mask)
            intent_predictions = torch.argmax(intent_logits, dim=-1)
            
            # Store results
            all_predictions.append(intent_predictions.cpu().numpy())
            all_input_ids.append(input_ids.cpu().numpy())
            all_attention_masks.append(attention_mask.cpu().numpy())
            all_true_slots.append(slot_labels.numpy())
    
    print(f"✅ Extracted {len(all_predictions)} batches of intent predictions")
    return all_predictions, all_input_ids, all_attention_masks, all_true_slots


def create_pipeline_data_loaders(data, intent_model, device, batch_size=32):
    """
    Create data loaders for pipeline training using intent predictions.
    
    Returns:
        train_loader: DataLoader with (input_ids, intent_predictions, slot_labels)
        test_loader: DataLoader with (input_ids, intent_predictions, slot_labels)
    """
    X_train, X_test = data['X_train'], data['X_test']
    y_slots_train, y_slots_test = data['y_slots_train'], data['y_slots_test']
    
    # Create attention masks
    train_attention_mask = create_attention_mask(X_train)
    test_attention_mask = create_attention_mask(X_test)
    
    # Create temporary data loaders to get intent predictions
    X_train_tensor = torch.LongTensor(X_train)
    X_test_tensor = torch.LongTensor(X_test)
    y_slots_train_tensor = torch.LongTensor(y_slots_train)
    y_slots_test_tensor = torch.LongTensor(y_slots_test)
    train_mask_tensor = torch.FloatTensor(train_attention_mask)
    test_mask_tensor = torch.FloatTensor(test_attention_mask)
    
    # Temporary data loaders for intent prediction extraction
    temp_train_dataset = TensorDataset(X_train_tensor, train_mask_tensor, y_slots_train_tensor)
    temp_test_dataset = TensorDataset(X_test_tensor, test_mask_tensor, y_slots_test_tensor)
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=batch_size, shuffle=False)
    temp_test_loader = DataLoader(temp_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract intent predictions
    train_intent_preds, train_input_ids, train_masks, train_slots = extract_intent_predictions(
        intent_model, temp_train_loader, device)
    test_intent_preds, test_input_ids, test_masks, test_slots = extract_intent_predictions(
        intent_model, temp_test_loader, device)
    
    # Reconstruct full arrays
    train_input_ids_full = np.concatenate(train_input_ids, axis=0)
    train_intent_preds_full = np.concatenate(train_intent_preds, axis=0)
    train_masks_full = np.concatenate(train_masks, axis=0)
    train_slots_full = np.concatenate(train_slots, axis=0)
    
    test_input_ids_full = np.concatenate(test_input_ids, axis=0)
    test_intent_preds_full = np.concatenate(test_intent_preds, axis=0)
    test_masks_full = np.concatenate(test_masks, axis=0)
    test_slots_full = np.concatenate(test_slots, axis=0)
    
    # Create final tensors
    train_input_tensor = torch.LongTensor(train_input_ids_full)
    train_intent_tensor = torch.LongTensor(train_intent_preds_full)
    train_mask_tensor = torch.FloatTensor(train_masks_full)
    train_slot_tensor = torch.LongTensor(train_slots_full)
    
    test_input_tensor = torch.LongTensor(test_input_ids_full)
    test_intent_tensor = torch.LongTensor(test_intent_preds_full)
    test_mask_tensor = torch.FloatTensor(test_masks_full)
    test_slot_tensor = torch.LongTensor(test_slots_full)
    
    # Create final datasets
    train_dataset = TensorDataset(train_input_tensor, train_intent_tensor, train_mask_tensor, train_slot_tensor)
    test_dataset = TensorDataset(test_input_tensor, test_intent_tensor, test_mask_tensor, test_slot_tensor)
    
    # Create final data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Created pipeline data loaders:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


def train_pipeline_model(pipeline_model, train_loader, val_loader, num_epochs=15, lr=1e-4, 
                        device='cuda', save_checkpoints=True, checkpoint_dir="checkpoints", 
                        model_name="pipeline_model"):
    """Train the intent-to-slot pipeline model."""
    pipeline_model.to(device)
    
    # Initialize weights properly
    init_model_weights(pipeline_model)
    
    optimizer = torch.optim.Adam(pipeline_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = padding index for slots
    
    print(f"🔄 Training {pipeline_model.__class__.__name__} Pipeline Model...")
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
            pipeline_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['best_f1']
            training_history = checkpoint['training_history']
            print(f"✅ Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        pipeline_model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (input_ids, intent_predictions, attention_mask, slot_labels) in enumerate(pbar):
                input_ids = input_ids.to(device)
                intent_predictions = intent_predictions.to(device)
                attention_mask = attention_mask.to(device)
                slot_labels = slot_labels.to(device)
                
                optimizer.zero_grad()
                
                logits = pipeline_model(input_ids, intent_predictions, attention_mask)
                
                # Reshape for loss calculation (slot filling is sequence labeling)
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = slot_labels.view(-1)
                
                loss = criterion(logits_flat, labels_flat)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(pipeline_model.parameters(), max_norm=1.0)
                
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
        pipeline_model.eval()
        val_metrics = evaluate_pipeline_model(pipeline_model, val_loader, device, verbose=False)
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
                'model_state_dict': pipeline_model.state_dict(),
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
                    'model_state_dict': pipeline_model.state_dict(),
                    'val_metrics': val_metrics,
                    'training_history': training_history
                }, best_checkpoint)
                print(f"💾 New best model saved! F1: {best_f1:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                periodic_checkpoint = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pipeline_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, periodic_checkpoint)
    
    print(f"✅ Pipeline training completed! Best F1: {best_f1:.4f}")
    return pipeline_model, training_history


def evaluate_pipeline_model(pipeline_model, test_loader, device, verbose=True):
    """Evaluate the intent-to-slot pipeline model."""
    pipeline_model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    with torch.no_grad():
        for input_ids, intent_predictions, attention_mask, slot_labels in test_loader:
            input_ids = input_ids.to(device)
            intent_predictions = intent_predictions.to(device)
            attention_mask = attention_mask.to(device)
            slot_labels = slot_labels.to(device)
            
            logits = pipeline_model(input_ids, intent_predictions, attention_mask)
            
            # Calculate loss
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = slot_labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels (excluding padding tokens)
            for i in range(batch_size):
                mask = attention_mask[i].bool()
                seq_predictions = predictions[i][mask].cpu().numpy()
                seq_labels = slot_labels[i][mask].cpu().numpy()
                
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
            print(f"🎯 Intent→Slot Pipeline Model Evaluation:")
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


# ============================================================
# Pipeline Experiment Runner
# ============================================================

def load_or_train_intent_model(data, vocabularies, model_type='LSTM', dataset_name='atis', device='cuda'):
    """
    Load a pre-trained intent model or train one if not available.
    
    Returns:
        intent_model: Trained intent classification model
        intent_metrics: Performance metrics of the intent model
    """
    print(f"\n{'='*50}")
    print(f"🔄 Loading/Training {model_type} Intent Model for Pipeline")
    print(f"{'='*50}")
    
    # Model configuration
    vocab_size = len(vocabularies['word2idx'])
    num_intent_labels = len(vocabularies['intent2idx'])
    
    if model_type == 'LSTM':
        intent_model = LSTMForIntentClassification(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_size=128,
            num_labels=num_intent_labels,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        model_class_name = 'LSTM'
    else:
        intent_model = RNNForIntentClassification(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_size=128,
            num_labels=num_intent_labels,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        model_class_name = 'RNN'
    
    # Try to load existing checkpoint
    checkpoint_name = f"{model_class_name}_intent_{dataset_name}"
    checkpoint_path = os.path.join("experiments/checkpoints", checkpoint_name, "best_model.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"📁 Loading pre-trained intent model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        intent_model.load_state_dict(checkpoint['model_state_dict'])
        intent_model.to(device)
        
        # Get metrics if available
        intent_metrics = checkpoint.get('val_metrics', {'f1': 0.0, 'accuracy': 0.0})
        print(f"✅ Loaded intent model with F1: {intent_metrics['f1']:.4f}, Acc: {intent_metrics['accuracy']:.4f}")
        
    else:
        print(f"❌ No pre-trained intent model found at: {checkpoint_path}")
        print("🔄 Training new intent model...")
        
        # Import training function
        from experiments.training_experiment1_fixed import train_intent_classification, create_data_loaders
        
        # Create data loaders for intent training
        train_loader, test_loader = create_data_loaders(data, 'intent', batch_size=32)
        
        # Train the intent model
        intent_model, history = train_intent_classification(
            intent_model, train_loader, test_loader,
            num_epochs=15, lr=1e-4, device=device,
            save_checkpoints=True, checkpoint_dir="experiments/checkpoints",
            model_name=checkpoint_name
        )
        
        # Evaluate the intent model
        intent_metrics = evaluate_intent_classification(intent_model, test_loader, device)
        print(f"✅ Trained intent model with F1: {intent_metrics['f1']:.4f}, Acc: {intent_metrics['accuracy']:.4f}")
    
    return intent_model, intent_metrics


def run_intent_to_slot_pipeline_experiment(dataset_name='atis', device='cuda'):
    """Run the complete intent-to-slot pipeline experiment."""
    print(f"\n{'='*60}")
    print(f"🚀 Intent → Slot Pipeline Experiment - {dataset_name.upper()}")
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
    
    print(f"📊 Dataset info:")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Intent labels: {num_intent_labels}")
    print(f"   Slot labels: {num_slot_labels}")
    
    results = {}
    
    # Train both RNN and LSTM pipeline models
    for model_type in ['RNN', 'LSTM']:
        print(f"\n{'='*50}")
        print(f"🔄 Training {model_type} Pipeline Model")
        print(f"{'='*50}")
        
        try:
            # Step 1: Load or train intent model
            intent_model, intent_metrics = load_or_train_intent_model(
                data, vocabularies, model_type, dataset_name, device)
            
            # Step 2: Create pipeline data loaders with intent predictions
            train_loader, test_loader = create_pipeline_data_loaders(
                data, intent_model, device, batch_size=32)
            
            # Step 3: Create pipeline model
            pipeline_model = IntentToSlotPipelineModel(
                vocab_size=vocab_size,
                num_intent_labels=num_intent_labels,
                num_slot_labels=num_slot_labels,
                embedding_dim=128,
                intent_embedding_dim=32,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                bidirectional=True,
                model_type=model_type
            )
            
            # Step 4: Train pipeline model
            checkpoint_name = f"{model_type}_pipeline_intent_to_slot_{dataset_name}"
            trained_pipeline_model, history = train_pipeline_model(
                pipeline_model, train_loader, test_loader,
                num_epochs=15, lr=1e-4, device=device,
                save_checkpoints=True, checkpoint_dir="experiments/checkpoints",
                model_name=checkpoint_name
            )
            
            # Step 5: Final evaluation
            final_metrics = evaluate_pipeline_model(trained_pipeline_model, test_loader, device)
            
            # Store results
            results[f"{model_type}_pipeline"] = {
                'model': f"{model_type}_Pipeline",
                'task': 'intent_to_slot',
                'dataset': dataset_name,
                'intent_metrics': intent_metrics,
                'pipeline_metrics': final_metrics,
                'history': history,
                'config': {
                    'vocab_size': vocab_size,
                    'num_intent_labels': num_intent_labels,
                    'num_slot_labels': num_slot_labels,
                    'embedding_dim': 128,
                    'intent_embedding_dim': 32,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': True,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'num_epochs': 15
                }
            }
            
            print(f"✅ {model_type} pipeline training completed!")
            print(f"🏆 Pipeline metrics:")
            print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"   Precision: {final_metrics['precision']:.4f}")
            print(f"   Recall: {final_metrics['recall']:.4f}")
            print(f"   F1-score: {final_metrics['f1']:.4f}")
            print(f"📊 Intent model metrics:")
            print(f"   Accuracy: {intent_metrics['accuracy']:.4f}")
            print(f"   Precision: {intent_metrics['precision']:.4f}")
            print(f"   Recall: {intent_metrics['recall']:.4f}")
            print(f"   F1-score: {intent_metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to train {model_type} pipeline: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def quick_pipeline_test():
    """Quick test of the intent-to-slot pipeline architecture."""
    print("🧪 Running quick intent→slot pipeline test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data, vocabularies = load_dataset('atis')
    
    # Create a simple intent model for testing
    intent_model = RNNForIntentClassification(
        vocab_size=len(vocabularies['word2idx']),
        embedding_dim=64,
        hidden_size=64,
        num_labels=len(vocabularies['intent2idx']),
        num_layers=1,
        dropout=0.1,
        bidirectional=False
    )
    intent_model.to(device)
    init_model_weights(intent_model)
    
    # Create pipeline data loaders
    train_loader, test_loader = create_pipeline_data_loaders(data, intent_model, device, batch_size=8)
    
    # Create pipeline model
    pipeline_model = IntentToSlotPipelineModel(
        vocab_size=len(vocabularies['word2idx']),
        num_intent_labels=len(vocabularies['intent2idx']),
        num_slot_labels=len(vocabularies['slot2idx']),
        embedding_dim=64,
        intent_embedding_dim=16,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        model_type='RNN'
    )
    
    # Test forward pass
    for input_ids, intent_predictions, attention_mask, slot_labels in test_loader:
        input_ids = input_ids.to(device)
        intent_predictions = intent_predictions.to(device)
        attention_mask = attention_mask.to(device)
        
        logits = pipeline_model(input_ids, intent_predictions, attention_mask)
        print(f"✅ Intent→Slot pipeline forward pass successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Intent predictions shape: {intent_predictions.shape}")
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Expected: [batch_size, seq_len, num_slot_labels]")
        break
    
    print("✅ Quick intent→slot pipeline test completed!")


def main():
    """Main function for intent-to-slot pipeline experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"\n{'='*70}")
    print("🚀 INTENT → SLOT PIPELINE EXPERIMENT")
    print(f"{'='*70}")
    print("Pipeline architecture:")
    print("✅ Step 1: Train/load intent classification model")
    print("✅ Step 2: Extract intent predictions on slot filling data")
    print("✅ Step 3: Train slot model with [word embeddings + intent embeddings]")
    print("✅ Step 4: Compare with independent models")
    print("")
    
    # Run quick test first
    print("Running quick test to verify pipeline architecture...")
    quick_pipeline_test()
    
    print("\n" + "="*50)
    print("Quick test passed! Ready for full pipeline experiment.")
    print("Uncomment the lines below to run full experiment:")
    print("")
    
    # Full experiment (uncomment to run)
    all_results = {}
    for dataset in ['atis', 'slurp']:
        print(f"\n🎯 Starting {dataset.upper()} intent→slot pipeline experiment...")
        try:
            results = run_intent_to_slot_pipeline_experiment(dataset, device)
            all_results[dataset] = results
            print(f"✅ {dataset.upper()} intent→slot pipeline experiment completed!")
        except Exception as e:
            print(f"❌ Failed pipeline experiment for {dataset}: {e}")
    
    # Print comprehensive results comparison
    print(f"\n🏆 COMPREHENSIVE INTENT→SLOT PIPELINE RESULTS")
    print("=" * 120)
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()} Dataset Pipeline Results:")
        print("-" * 80)
        print(f"{'Model':<15} {'Task':<15} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        for key, result in results.items():
            # Show intent model metrics
            intent_metrics = result['intent_metrics']
            print(f"{result['model']:<15} {'Intent Class':<15} {intent_metrics['accuracy']:<10.4f} {intent_metrics['precision']:<12.4f} {intent_metrics['recall']:<10.4f} {intent_metrics['f1']:<10.4f}")
            
            # Show pipeline metrics
            pipeline_metrics = result['pipeline_metrics']
            print(f"{result['model']:<15} {'Pipeline Slot':<15} {pipeline_metrics['accuracy']:<10.4f} {pipeline_metrics['precision']:<12.4f} {pipeline_metrics['recall']:<10.4f} {pipeline_metrics['f1']:<10.4f}")
            print()
    print("=" * 120)


if __name__ == "__main__":
    main()