"""
atis_preprocess.py
------------------
Driver for preprocessing ATIS dataset using preprocess_utils.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess_utils import load_atis_dataset, preprocess_data, save_preprocessed_data

def main():
    """
    Main function to preprocess ATIS dataset using unified utilities.
    """
    print("="*70)
    print("ATIS DATASET PREPROCESSING")
    print("="*70)
    
    # Load ATIS dataset
    data = load_atis_dataset()
    
    # Build vocab on train and reuse for test
    print("\n>>> Preprocessing ATIS data...")
    X_train, y_slots_train, y_intents_train, word2idx, slot2idx, intent2idx, max_len = preprocess_data(data["train"])
    X_test, y_slots_test, y_intents_test, _, _, _, _ = preprocess_data(
        data["test"], word2idx, slot2idx, intent2idx, max_len
    )
    
    print(f"  >>> Train shape: {X_train.shape}")
    print(f"  >>> Test shape: {X_test.shape}")
    print(f"  >>> Max sequence length: {max_len}")
    print(f"  >>> Vocabulary size: {len(word2idx)}")
    print(f"  >>> Slot labels: {len(slot2idx)}")
    print(f"  >>> Intent labels: {len(intent2idx)}")
    
    # Prepare data dictionary
    processed_data = {
        "train": (X_train, y_slots_train, y_intents_train),
        "test": (X_test, y_slots_test, y_intents_test)
    }
    
    # Save preprocessed data
    save_preprocessed_data(
        dataset_name="atis",
        data_dict=processed_data,
        vocabularies=(word2idx, slot2idx, intent2idx),
        max_len=max_len
    )
    
    print("\n" + "="*70)
    print(">>> ATIS preprocessing complete!")
    print("="*70)
    
    return processed_data, (word2idx, slot2idx, intent2idx), max_len

if __name__ == "__main__":
    main()