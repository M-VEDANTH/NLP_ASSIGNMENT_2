
# Slot Filling and Intent Classification Project

This project implements and compares RNN and LSTM architectures for slot filling and intent classification tasks on ATIS and SLURP datasets.

## Project Structure
```
assign_2/
├── data/               # Dataset storage and preprocessing
├── models/             # RNN and LSTM model implementations
├── experiments/        # Four experimental setups
├── utils/              # Utility functions and evaluation metrics
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Datasets
1. **ATIS**: Airline Travel Information System dataset for slot filling and intent classification
2. **SLURP**: Spoken Language Understanding Resource Package

## Experimental Setup
1. **Independent Models**: Separate slot filling and intent classification
2. **Slot → Intent**: Use slot predictions for intent classification
3. **Intent → Slot**: Use intent predictions for slot filling  
4. **Joint Multi-Task**: Shared encoder with dual output heads

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run data preprocessing: `python utils/preprocess_data.py`
3. Train models: `python experiments/[experiment_file_name].py`

## Evaluation Metrics
- Precision, Recall, F1-score
- Accuracy
## Results (Drive Link):
- https://docs.google.com/document/d/1IVmSqYH3X7bYkDMxIgjzm0L_beupH7di9pMsSOI1GnQ/edit?usp=sharing

