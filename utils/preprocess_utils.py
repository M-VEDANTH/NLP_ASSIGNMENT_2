"""
preprocess_utils.py
-------------------
Common preprocessing utilities for ATIS and SLURP datasets.

Functions:
- build_vocab
- pad_sequences
- preprocess_data
- load_atis_dataset
- load_slurp_dataset
"""

import json
import numpy as np
import os
from collections import Counter
from datasets import load_dataset

# ------------------------------
# Vocabulary helpers
# ------------------------------

def build_vocab(sequences, add_pad=True, add_unk=True):
    """
    Build token-to-index and index-to-token mappings.
    """
    tokens = sorted({tok for seq in sequences for tok in seq})
    word2idx, idx2word = {}, {}
    idx = 0

    if add_pad:
        word2idx["PAD"] = idx; idx2word[idx] = "PAD"; idx += 1
    if add_unk:
        word2idx["UNK"] = idx; idx2word[idx] = "UNK"; idx += 1

    for t in tokens:
        if t not in ("PAD", "UNK"):
            word2idx[t] = idx
            idx2word[idx] = t
            idx += 1

    return word2idx, idx2word


def pad_sequences(sequences, pad_value=0, max_len=None):
    """
    Pads or truncates all sequences to the same length.
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            # Truncate if too long
            padded.append(seq[:max_len])
        else:
            # Pad if too short
            padded.append(seq + [pad_value] * (max_len - len(seq)))
    
    return np.array(padded), max_len


# ------------------------------
# Generic preprocessing pipeline
# ------------------------------

def preprocess_data(dataset, word2idx=None, slot2idx=None, intent2idx=None, max_len=None):
    """
    Convert dataset (list of dicts with 'tokens', 'slots', 'intent') to numeric arrays.
    """

    sentences = [d["tokens"] for d in dataset]
    slots = [d["slots"] for d in dataset]
    intents = [d["intent"] for d in dataset]

    # Build vocabs if not provided
    if word2idx is None:
        word2idx, _ = build_vocab(sentences)
    if slot2idx is None:
        slot2idx, _ = build_vocab(slots)
    if intent2idx is None:
        intent2idx, _ = build_vocab([[i] for i in intents], add_pad=False, add_unk=False)

    # Encode
    X = [[word2idx.get(tok, word2idx.get("UNK", 0)) for tok in seq] for seq in sentences]
    y_slots = [[slot2idx.get(tag, slot2idx.get("PAD", 0)) for tag in seq] for seq in slots]
    y_intents = [intent2idx.get(i, 0) for i in intents]

    # Pad
    X_pad, max_len = pad_sequences(X, pad_value=word2idx["PAD"], max_len=max_len)
    y_slots_pad, _ = pad_sequences(y_slots, pad_value=slot2idx["PAD"], max_len=max_len)
    y_intents = np.array(y_intents)

    return X_pad, y_slots_pad, y_intents, word2idx, slot2idx, intent2idx, max_len


# ------------------------------
# ATIS dataset loader
# ------------------------------

def load_atis_dataset():
    """
    Load ATIS dataset from Hugging Face and convert into our format.
    """
    print(">>> Loading ATIS dataset from Hugging Face...")
    data = load_dataset("tuetschek/atis")

    def convert(split):
        result = []
        for sample in data[split]:
            tokens = sample["text"]
            if isinstance(tokens, str):
                tokens = tokens.split()
            
            slots = sample["slots"]
            if isinstance(slots, str):
                slots = slots.split()
                
            intent = sample["intent"]
            result.append({"tokens": tokens, "slots": slots, "intent": intent})
        return result

    atis_data = {
        "train": convert("train"),
        "test": convert("test"),  # ATIS doesn't have validation in this format
    }
    
    print(f"  >>> Train: {len(atis_data['train'])} samples")
    print(f"  >>> Test: {len(atis_data['test'])} samples")
    
    return atis_data


# ------------------------------
# SLURP dataset loader
# ------------------------------

def load_slurp_dataset(base_path):
    """
    Load SLURP dataset (expects train.jsonl, devel.jsonl, test.jsonl under base_path)
    """
    print(f">>> Loading SLURP dataset from {base_path}...")
    
    def load_split(path, split_name):
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping {split_name}")
            return []
            
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Handle token field
                tokens_field = item.get("tokens")
                if tokens_field and isinstance(tokens_field[0], dict):
                    tokens = [t.get("surface") or t.get("text") for t in tokens_field]
                else:
                    tokens = tokens_field or item["sentence"].split()
                
                # Slots
                slots = item.get("slots")
                # If not provided, derive from entities
                if not slots:
                    slots = ["O"] * len(tokens)
                    for ent in item.get("entities", []):
                        span = ent.get("span") or []
                        for i, tok_idx in enumerate(span):
                            if tok_idx < len(slots):
                                slots[tok_idx] = f"{'B' if i==0 else 'I'}-{ent.get('type','')}"
                
                intent = item["intent"]
                data.append({"tokens": tokens, "slots": slots, "intent": intent})
        return data

    slurp_data = {
        "train": load_split(f"{base_path}/train.jsonl", "train"),
        "validation": load_split(f"{base_path}/devel.jsonl", "validation"),
        "test": load_split(f"{base_path}/test.jsonl", "test"),
    }
    
    print(f"  >>> Train: {len(slurp_data['train'])} samples")
    print(f"  >>> Validation: {len(slurp_data['validation'])} samples")
    print(f"  >>> Test: {len(slurp_data['test'])} samples")
    
    return slurp_data


# ------------------------------
# Unified save function
# ------------------------------

def save_preprocessed_data(dataset_name, data_dict, vocabularies, max_len, output_dir="data"):
    """
    Save preprocessed data in both .npz and individual .npy formats for consistency.
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f">>> Saving {dataset_name} data to {dataset_dir}...")
    
    # Save as .npz (compact format)
    npz_arrays = {}
    npy_files = []
    
    for split, arrays in data_dict.items():
        if split in ['train', 'validation', 'test']:
            X, y_slots, y_intents = arrays
            
            # Add to npz
            npz_arrays[f"X_{split}"] = X
            npz_arrays[f"y_slots_{split}"] = y_slots
            npz_arrays[f"y_intents_{split}"] = y_intents
            
            # Save individual .npy files
            np.save(os.path.join(dataset_dir, f"X_{split}.npy"), X)
            np.save(os.path.join(dataset_dir, f"y_slots_{split}.npy"), y_slots)
            np.save(os.path.join(dataset_dir, f"y_intents_{split}.npy"), y_intents)
            
            npy_files.extend([f"X_{split}.npy", f"y_slots_{split}.npy", f"y_intents_{split}.npy"])
    
    # Save .npz file
    np.savez(os.path.join(dataset_dir, f"{dataset_name}_data.npz"), **npz_arrays)
    
    # Save vocabularies
    vocab_data = {
        "word2idx": vocabularies[0],
        "slot2idx": vocabularies[1], 
        "intent2idx": vocabularies[2],
        "max_len": max_len,
        # Also include list format for consistency
        "word_vocab": list(vocabularies[0].keys()),
        "slot_vocab": list(vocabularies[1].keys()),
        "intent_vocab": list(vocabularies[2].keys())
    }
    
    with open(os.path.join(dataset_dir, f"{dataset_name}_vocabularies.json"), "w") as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"  >>> Saved {dataset_name}_data.npz")
    print(f"  >>> Saved {dataset_name}_vocabularies.json") 
    for file in npy_files:
        print(f"  >>> Saved {file}")
    
    return dataset_dir