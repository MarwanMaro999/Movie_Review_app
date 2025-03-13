# Movie Review Sentiment Analysis

## Project Overview
This project implements a deep learning model to analyze movie reviews and predict whether they are positive or negative. The model uses LSTM (Long Short-Term Memory) neural networks to understand the sentiment behind movie reviews, achieving approximately 84% accuracy on the validation set.

## Features
- Text preprocessing pipeline including tokenization, stop word removal, and lemmatization
- LSTM-based neural network architecture
- Binary classification (positive/negative reviews)
- Pre-trained model and tokenizer for immediate use
- Support for real-time prediction of new reviews

## Technical Details

### Data Processing
- Removal of special characters and numbers
- Text normalization and lemmatization
- Stop words removal
- Sequence padding for consistent input size

### Model Architecture
- Embedding layer for word vector representation
- LSTM layer with L2 regularization
- Dropout layer (0.7) for preventing overfinding
- Dense layer with sigmoid activation for binary classification

### Performance
- Training Accuracy: ~85%
- Validation Accuracy: ~84%
- Loss: ~0.39

## Requirements
```
tensorflow
numpy
pandas
nltk
scikit-learn
keras
```

## Model Training Process
1. Data preprocessing and cleaning
2. Text tokenization and sequence padding
3. Model architecture design with embedding and LSTM layers
4. Training with early stopping to prevent overfitting
5. Model evaluation and testing

# Project Video
https://github.com/user-attachments/assets/e5d5a06f-07b9-4826-b5d7-7457de30a85c
