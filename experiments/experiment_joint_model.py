"""
Experiment 4: Joint Model with Shared Encoder
--------------------------------------------
Multi-task learning experiment where slot filling and intent classification 
are trained simultaneously using a shared encoder.

Architecture:
1. Shared encoder (RNN/LSTM) processes input text
2. Task-specific heads for slot filling and intent classification
3. Joint training with combined multi-task loss
4. Compare with independent and pipeline models

The joint model works as follows:
- Input text → Shared Encoder → Encoded representations
- Encoded representations → Slot Head → Slot predictions
- Encoded representations → Intent Head → Intent prediction
- Loss = α * slot_loss + β * intent_loss
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
    load_dataset, create_attention_mask
)


# ============================================================
# Joint Model Architecture
# ============================================================

class JointSlotIntentModel(nn.Module):
    """
    Joint model for slot filling and intent classification with shared encoder.
    
    Architecture:
    - Shared encoder (RNN/LSTM) 
    - Task-specific heads for slot filling (sequence labeling)
    - Task-specific heads for intent classification
    """
    
    def __init__(self, vocab_size, num_slot_labels, num_intent_labels, 
                 embedding_dim=128, hidden_size=128, num_layers=2, 
                 dropout=0.3, bidirectional=True, model_type='LSTM'):
        super(JointSlotIntentModel, self).__init__()
        
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared components
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Shared encoder
        if model_type == 'LSTM':
            self.shared_encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:  # RNN
            self.shared_encoder = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        # Output size adjustment for bidirectional
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Task-specific components
        self.shared_dropout = nn.Dropout(dropout)
        
        # Slot filling head (sequence labeling)
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_slot_labels)
        )
        
        # Intent classification head
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_intent_labels)
        )
        
        print(f"✅ Created {model_type} Joint Model:")
        print(f"   Word embedding: {vocab_size} → {embedding_dim}")
        print(f"   Shared encoder: {embedding_dim} → {encoder_output_size}")
        print(f"   Slot head: {encoder_output_size} → {num_slot_labels}")
        print(f"   Intent head: {encoder_output_size} → {num_intent_labels}")
        print(f"   Bidirectional: {bidirectional}, Layers: {num_layers}")
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for joint model.
        
        Args:
            input_ids: [batch_size, seq_len] - word token ids
            attention_mask: [batch_size, seq_len] - attention mask
            
        Returns:
            slot_logits: [batch_size, seq_len, num_slot_labels] - slot predictions
            intent_logits: [batch_size, num_intent_labels] - intent predictions
        """
        batch_size, seq_len = input_ids.size()
        
        # Get word embeddings
        word_embeddings = self.word_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply attention mask to embeddings if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(word_embeddings)
            word_embeddings = word_embeddings * mask_expanded
        
        # Pass through shared encoder
        encoder_output, hidden = self.shared_encoder(word_embeddings)
        # encoder_output: [batch_size, seq_len, hidden_size * num_directions]
        
        # Apply shared dropout
        encoder_output = self.shared_dropout(encoder_output)
        
        # Slot filling head (sequence labeling)
        slot_logits = self.slot_head(encoder_output)
        # slot_logits: [batch_size, seq_len, num_slot_labels]
        
        # Intent classification head (use pooled representation)
        if self.bidirectional:
            # For bidirectional, concatenate forward and backward final hidden states
            if self.model_type == 'LSTM':
                h_n = hidden[0]  # hidden state
            else:
                h_n = hidden
            
            # h_n: [num_layers * num_directions, batch_size, hidden_size]
            # Get the last layer's hidden states
            forward_hidden = h_n[-2, :, :]  # Forward direction of last layer
            backward_hidden = h_n[-1, :, :] # Backward direction of last layer
            pooled_output = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # For unidirectional, use the last hidden state
            if attention_mask is not None:
                # Get the last valid position for each sequence
                lengths = (attention_mask.sum(dim=1) - 1).long()  # Convert to long
                batch_size = encoder_output.size(0)
                pooled_output = encoder_output[range(batch_size), lengths]
            else:
                pooled_output = encoder_output[:, -1, :]
        
        intent_logits = self.intent_head(pooled_output)
        # intent_logits: [batch_size, num_intent_labels]
        
        return slot_logits, intent_logits


# ============================================================
# Joint Training Functions
# ============================================================

def create_joint_data_loaders(data, batch_size=32, shuffle=True):
    """Create DataLoader for joint training with both slot and intent labels."""
    X_train, X_test = data['X_train'], data['X_test']
    y_slots_train, y_slots_test = data['y_slots_train'], data['y_slots_test']
    y_intents_train, y_intents_test = data['y_intents_train'], data['y_intents_test']
    
    # Create attention masks
    train_attention_mask = create_attention_mask(X_train)
    test_attention_mask = create_attention_mask(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.LongTensor(X_train)
    X_test_tensor = torch.LongTensor(X_test)
    y_slots_train_tensor = torch.LongTensor(y_slots_train)
    y_slots_test_tensor = torch.LongTensor(y_slots_test)
    y_intents_train_tensor = torch.LongTensor(y_intents_train)
    y_intents_test_tensor = torch.LongTensor(y_intents_test)
    train_mask_tensor = torch.FloatTensor(train_attention_mask)
    test_mask_tensor = torch.FloatTensor(test_attention_mask)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, train_mask_tensor, y_slots_train_tensor, y_intents_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, test_mask_tensor, y_slots_test_tensor, y_intents_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Created joint data loaders:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


def train_joint_model(joint_model, train_loader, val_loader, num_epochs=15, lr=1e-4, 
                     slot_weight=1.0, intent_weight=1.0, device='cuda', 
                     save_checkpoints=True, checkpoint_dir="checkpoints", 
                     model_name="joint_model"):
    """
    Train the joint model with multi-task loss.
    
    Args:
        slot_weight: Weight for slot filling loss
        intent_weight: Weight for intent classification loss
    """
    joint_model.to(device)
    
    # Initialize weights properly
    init_model_weights(joint_model)
    
    optimizer = torch.optim.Adam(joint_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    
    # Loss functions
    slot_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = padding index for slots
    intent_criterion = nn.CrossEntropyLoss()
    
    print(f"🔄 Training {joint_model.__class__.__name__} Joint Model...")
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"🎯 Learning rate: {lr}, Weight decay: 1e-5")
    print(f"⚖️ Loss weights: Slot={slot_weight}, Intent={intent_weight}")

    # Create checkpoint directory
    if save_checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"💾 Checkpoints will be saved to: {checkpoint_path}")

    best_combined_score = 0.0
    training_history = {
        'train_loss': [], 'slot_loss': [], 'intent_loss': [],
        'val_slot_f1': [], 'val_slot_acc': [],
        'val_intent_f1': [], 'val_intent_acc': [],
        'combined_score': [], 'learning_rates': []
    }
    start_epoch = 0
    
    # Try to load existing checkpoint
    if save_checkpoints:
        latest_checkpoint = os.path.join(checkpoint_path, 'latest_checkpoint.pt')
        if os.path.exists(latest_checkpoint):
            print("🔄 Found existing checkpoint, loading...")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            joint_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_combined_score = checkpoint['best_combined_score']
            training_history = checkpoint['training_history']
            print(f"✅ Resumed from epoch {start_epoch}, best combined score: {best_combined_score:.4f}")

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        joint_model.train()
        total_loss = 0
        slot_loss_sum = 0
        intent_loss_sum = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (input_ids, attention_mask, slot_labels, intent_labels) in enumerate(pbar):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                slot_labels = slot_labels.to(device)
                intent_labels = intent_labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                slot_logits, intent_logits = joint_model(input_ids, attention_mask)
                
                # Calculate slot loss (sequence labeling)
                batch_size, seq_len, num_slot_classes = slot_logits.shape
                slot_logits_flat = slot_logits.view(-1, num_slot_classes)
                slot_labels_flat = slot_labels.view(-1)
                slot_loss = slot_criterion(slot_logits_flat, slot_labels_flat)
                
                # Calculate intent loss
                intent_loss = intent_criterion(intent_logits, intent_labels)
                
                # Combined loss
                combined_loss = slot_weight * slot_loss + intent_weight * intent_loss
                
                combined_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(joint_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += combined_loss.item()
                slot_loss_sum += slot_loss.item()
                intent_loss_sum += intent_loss.item()
                num_batches += 1
                
                # Enhanced progress info
                if batch_idx == 0 or (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                    pbar.set_postfix({
                        'total': f'{combined_loss.item():.4f}',
                        'slot': f'{slot_loss.item():.4f}',
                        'intent': f'{intent_loss.item():.4f}'
                    })
        
        avg_train_loss = total_loss / num_batches
        avg_slot_loss = slot_loss_sum / num_batches
        avg_intent_loss = intent_loss_sum / num_batches
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['slot_loss'].append(avg_slot_loss)
        training_history['intent_loss'].append(avg_intent_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        joint_model.eval()
        val_metrics = evaluate_joint_model(joint_model, val_loader, device, verbose=False)
        
        # Store validation metrics
        training_history['val_slot_f1'].append(val_metrics['slot_metrics']['f1'])
        training_history['val_slot_acc'].append(val_metrics['slot_metrics']['accuracy'])
        training_history['val_intent_f1'].append(val_metrics['intent_metrics']['f1'])
        training_history['val_intent_acc'].append(val_metrics['intent_metrics']['accuracy'])
        
        # Combined score for model selection (average of F1 scores)
        combined_score = (val_metrics['slot_metrics']['f1'] + val_metrics['intent_metrics']['f1']) / 2
        training_history['combined_score'].append(combined_score)
        
        # Update learning rate scheduler
        scheduler.step(combined_score)
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}:")
        print(f"  Loss: {avg_train_loss:.4f} (Slot: {avg_slot_loss:.4f}, Intent: {avg_intent_loss:.4f})")
        print(f"  Slot: F1={val_metrics['slot_metrics']['f1']:.4f}, Acc={val_metrics['slot_metrics']['accuracy']:.4f}")
        print(f"  Intent: F1={val_metrics['intent_metrics']['f1']:.4f}, Acc={val_metrics['intent_metrics']['accuracy']:.4f}")
        print(f"  Combined: {combined_score:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        if save_checkpoints:
            # Save latest checkpoint
            latest_checkpoint = os.path.join(checkpoint_path, 'latest_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': joint_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_combined_score': best_combined_score,
                'training_history': training_history,
                'val_metrics': val_metrics
            }, latest_checkpoint)
            
            # Save best model
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_checkpoint = os.path.join(checkpoint_path, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': joint_model.state_dict(),
                    'val_metrics': val_metrics,
                    'training_history': training_history
                }, best_checkpoint)
                print(f"💾 New best model saved! Combined score: {best_combined_score:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                periodic_checkpoint = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': joint_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, periodic_checkpoint)
    
    print(f"✅ Joint training completed! Best combined score: {best_combined_score:.4f}")
    return joint_model, training_history


def evaluate_joint_model(joint_model, test_loader, device, verbose=True):
    """Evaluate the joint model on both tasks."""
    joint_model.eval()
    
    # Slot filling metrics
    slot_predictions = []
    slot_labels = []
    
    # Intent classification metrics
    intent_predictions = []
    intent_labels = []
    
    total_slot_loss = 0
    total_intent_loss = 0
    num_batches = 0
    
    slot_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    intent_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_ids, attention_mask, slot_targets, intent_targets in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            slot_targets = slot_targets.to(device)
            intent_targets = intent_targets.to(device)
            
            # Forward pass
            slot_logits, intent_logits = joint_model(input_ids, attention_mask)
            
            # Calculate losses
            batch_size, seq_len, num_slot_classes = slot_logits.shape
            slot_logits_flat = slot_logits.view(-1, num_slot_classes)
            slot_labels_flat = slot_targets.view(-1)
            slot_loss = slot_criterion(slot_logits_flat, slot_labels_flat)
            
            intent_loss = intent_criterion(intent_logits, intent_targets)
            
            total_slot_loss += slot_loss.item()
            total_intent_loss += intent_loss.item()
            num_batches += 1
            
            # Get predictions
            slot_preds = torch.argmax(slot_logits, dim=-1)
            intent_preds = torch.argmax(intent_logits, dim=-1)
            
            # Collect slot predictions and labels (excluding padding tokens)
            for i in range(batch_size):
                mask = attention_mask[i].bool()
                seq_slot_preds = slot_preds[i][mask].cpu().numpy()
                seq_slot_targets = slot_targets[i][mask].cpu().numpy()
                
                # Filter out padding tokens (0)
                non_pad_mask = seq_slot_targets != 0
                if non_pad_mask.sum() > 0:
                    slot_predictions.extend(seq_slot_preds[non_pad_mask])
                    slot_labels.extend(seq_slot_targets[non_pad_mask])
            
            # Collect intent predictions and labels
            intent_predictions.extend(intent_preds.cpu().numpy())
            intent_labels.extend(intent_targets.cpu().numpy())
    
    # Convert to numpy arrays
    slot_predictions = np.array(slot_predictions)
    slot_labels = np.array(slot_labels)
    intent_predictions = np.array(intent_predictions)
    intent_labels = np.array(intent_labels)
    
    # Calculate slot metrics
    if len(slot_predictions) > 0:
        slot_accuracy = accuracy_score(slot_labels, slot_predictions)
        slot_precision, slot_recall, slot_f1, _ = precision_recall_fscore_support(
            slot_labels, slot_predictions, average='weighted')
    else:
        slot_accuracy = slot_precision = slot_recall = slot_f1 = 0.0
    
    # Calculate intent metrics
    intent_accuracy = accuracy_score(intent_labels, intent_predictions)
    intent_precision, intent_recall, intent_f1, _ = precision_recall_fscore_support(
        intent_labels, intent_predictions, average='weighted')
    
    avg_slot_loss = total_slot_loss / num_batches
    avg_intent_loss = total_intent_loss / num_batches
    
    if verbose:
        print(f"🎯 Joint Model Evaluation:")
        print(f"📊 Slot Filling:")
        print(f"   Loss: {avg_slot_loss:.4f}")
        print(f"   Accuracy: {slot_accuracy:.4f}")
        print(f"   Precision: {slot_precision:.4f}")
        print(f"   Recall: {slot_recall:.4f}")
        print(f"   F1-score: {slot_f1:.4f}")
        print(f"   Samples: {len(slot_predictions)}")
        print(f"📊 Intent Classification:")
        print(f"   Loss: {avg_intent_loss:.4f}")
        print(f"   Accuracy: {intent_accuracy:.4f}")
        print(f"   Precision: {intent_precision:.4f}")
        print(f"   Recall: {intent_recall:.4f}")
        print(f"   F1-score: {intent_f1:.4f}")
        print(f"   Samples: {len(intent_predictions)}")
        
        # Combined metrics
        combined_f1 = (slot_f1 + intent_f1) / 2
        combined_acc = (slot_accuracy + intent_accuracy) / 2
        print(f"🏆 Combined:")
        print(f"   Average F1: {combined_f1:.4f}")
        print(f"   Average Accuracy: {combined_acc:.4f}")
    
    return {
        'slot_metrics': {
            'accuracy': slot_accuracy,
            'precision': slot_precision,
            'recall': slot_recall,
            'f1': slot_f1,
            'loss': avg_slot_loss,
            'num_samples': len(slot_predictions)
        },
        'intent_metrics': {
            'accuracy': intent_accuracy,
            'precision': intent_precision,
            'recall': intent_recall,
            'f1': intent_f1,
            'loss': avg_intent_loss,
            'num_samples': len(intent_predictions)
        },
        'combined_metrics': {
            'avg_f1': (slot_f1 + intent_f1) / 2,
            'avg_accuracy': (slot_accuracy + intent_accuracy) / 2
        }
    }


# ============================================================
# Joint Experiment Runner
# ============================================================

def run_joint_experiment(dataset_name='atis', device='cuda'):
    """Run the complete joint model experiment."""
    print(f"\n{'='*60}")
    print(f"🚀 Joint Model Experiment - {dataset_name.upper()}")
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
    print(f"   Slot labels: {num_slot_labels}")
    print(f"   Intent labels: {num_intent_labels}")
    
    results = {}
    
    # Train both RNN and LSTM joint models
    for model_type in ['RNN', 'LSTM']:
        print(f"\n{'='*50}")
        print(f"🔄 Training {model_type} Joint Model")
        print(f"{'='*50}")
        
        try:
            # Create joint data loaders
            train_loader, test_loader = create_joint_data_loaders(data, batch_size=32)
            
            # Create joint model
            joint_model = JointSlotIntentModel(
                vocab_size=vocab_size,
                num_slot_labels=num_slot_labels,
                num_intent_labels=num_intent_labels,
                embedding_dim=128,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                bidirectional=True,
                model_type=model_type
            )
            
            # Train joint model
            checkpoint_name = f"{model_type}_joint_{dataset_name}"
            trained_joint_model, history = train_joint_model(
                joint_model, train_loader, test_loader,
                num_epochs=15, lr=1e-4, 
                slot_weight=1.0, intent_weight=1.0,  # Equal weighting
                device=device,
                save_checkpoints=True, checkpoint_dir="experiments/checkpoints",
                model_name=checkpoint_name
            )
            
            # Final evaluation
            final_metrics = evaluate_joint_model(trained_joint_model, test_loader, device)
            
            # Store results
            results[f"{model_type}_joint"] = {
                'model': f"{model_type}_Joint",
                'task': 'joint_slot_intent',
                'dataset': dataset_name,
                'slot_metrics': final_metrics['slot_metrics'],
                'intent_metrics': final_metrics['intent_metrics'],
                'combined_metrics': final_metrics['combined_metrics'],
                'history': history,
                'config': {
                    'vocab_size': vocab_size,
                    'num_slot_labels': num_slot_labels,
                    'num_intent_labels': num_intent_labels,
                    'embedding_dim': 128,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': True,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'num_epochs': 15,
                    'slot_weight': 1.0,
                    'intent_weight': 1.0
                }
            }
            
            print(f"✅ {model_type} joint training completed!")
            print(f"🏆 Slot metrics:")
            print(f"   Accuracy: {final_metrics['slot_metrics']['accuracy']:.4f}")
            print(f"   Precision: {final_metrics['slot_metrics']['precision']:.4f}")
            print(f"   Recall: {final_metrics['slot_metrics']['recall']:.4f}")
            print(f"   F1-score: {final_metrics['slot_metrics']['f1']:.4f}")
            print(f"🏆 Intent metrics:")
            print(f"   Accuracy: {final_metrics['intent_metrics']['accuracy']:.4f}")
            print(f"   Precision: {final_metrics['intent_metrics']['precision']:.4f}")
            print(f"   Recall: {final_metrics['intent_metrics']['recall']:.4f}")
            print(f"   F1-score: {final_metrics['intent_metrics']['f1']:.4f}")
            print(f"🏆 Combined:")
            print(f"   Average F1: {final_metrics['combined_metrics']['avg_f1']:.4f}")
            print(f"   Average Accuracy: {final_metrics['combined_metrics']['avg_accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to train {model_type} joint model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def quick_joint_test():
    """Quick test of the joint model architecture."""
    print("🧪 Running quick joint model test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data, vocabularies = load_dataset('atis')
    
    # Create joint data loaders
    train_loader, test_loader = create_joint_data_loaders(data, batch_size=8)
    
    # Create joint model
    joint_model = JointSlotIntentModel(
        vocab_size=len(vocabularies['word2idx']),
        num_slot_labels=len(vocabularies['slot2idx']),
        num_intent_labels=len(vocabularies['intent2idx']),
        embedding_dim=64,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        model_type='RNN'
    )
    
    # Test forward pass
    for input_ids, attention_mask, slot_labels, intent_labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        slot_logits, intent_logits = joint_model(input_ids, attention_mask)
        print(f"✅ Joint model forward pass successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Slot logits shape: {slot_logits.shape}")
        print(f"   Intent logits shape: {intent_logits.shape}")
        print(f"   Expected slot: [batch_size, seq_len, num_slot_labels]")
        print(f"   Expected intent: [batch_size, num_intent_labels]")
        break
    
    print("✅ Quick joint model test completed!")


def main():
    """Main function for joint model experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"\n{'='*70}")
    print("🚀 JOINT MODEL WITH SHARED ENCODER EXPERIMENT")
    print(f"{'='*70}")
    print("Multi-task learning architecture:")
    print("✅ Shared encoder processes input text")
    print("✅ Task-specific heads for slot filling and intent classification")
    print("✅ Joint training with combined multi-task loss")
    print("✅ Compare with independent and pipeline models")
    print("")
    
    # Run quick test first
    print("Running quick test to verify joint architecture...")
    quick_joint_test()
    
    print("\n" + "="*50)
    print("Quick test passed! Ready for full joint experiment.")
    print("Uncomment the lines below to run full experiment:")
    print("")
    
    # Full experiment (uncomment to run)
    all_results = {}
    for dataset in ['atis', 'slurp']:
        print(f"\n🎯 Starting {dataset.upper()} joint experiment...")
        try:
            results = run_joint_experiment(dataset, device)
            all_results[dataset] = results
            print(f"✅ {dataset.upper()} joint experiment completed!")
        except Exception as e:
            print(f"❌ Failed joint experiment for {dataset}: {e}")
    
    # Print comprehensive results comparison
    print(f"\n🏆 COMPREHENSIVE JOINT MODEL RESULTS")
    print("=" * 140)
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()} Dataset Joint Results:")
        print("-" * 100)
        print(f"{'Model':<12} {'Task':<15} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Combined':<10}")
        print("-" * 100)
        for key, result in results.items():
            # Show slot metrics
            slot_metrics = result['slot_metrics']
            print(f"{result['model']:<12} {'Slot Filling':<15} {slot_metrics['accuracy']:<10.4f} {slot_metrics['precision']:<12.4f} {slot_metrics['recall']:<10.4f} {slot_metrics['f1']:<10.4f} {'-':<10}")
            
            # Show intent metrics
            intent_metrics = result['intent_metrics']
            print(f"{result['model']:<12} {'Intent Class':<15} {intent_metrics['accuracy']:<10.4f} {intent_metrics['precision']:<12.4f} {intent_metrics['recall']:<10.4f} {intent_metrics['f1']:<10.4f} {'-':<10}")
            
            # Show combined metrics
            combined_metrics = result['combined_metrics']
            print(f"{result['model']:<12} {'Combined':<15} {combined_metrics['avg_accuracy']:<10.4f} {'-':<12} {'-':<10} {combined_metrics['avg_f1']:<10.4f} {'★':<10}")
            print()
    print("=" * 140)


if __name__ == "__main__":
    main()