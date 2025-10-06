# NLP_ASSIGNMENT_2

Experimental variants: for each encoder type (LSTM and RNN) and each dataset (ATIS and SLURP) you must run:

Independent models: slot-only model and intent-only model trained separately.

Slot → Intent pipeline: run slot model first, use its predictions (or predicted distributions) as additional features for intent classifier.

Intent → Slot pipeline: run intent model first, use predicted intent as extra feature (per-token) for slot model.

Joint multi-task: shared encoder + two heads, train jointly with combined loss.
