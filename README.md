# Swahili News Classification using Deep Learning

## Overview
This project focuses on classifying Swahili news articles into predefined categories using deep learning models. Three different architectures have been implemented:

- **LSTM (Long Short-Term Memory) Model**
- **CNN (Convolutional Neural Network) Model**
- **Transformer-based Swahili BERT Model**

The goal is to provide an accurate classification model that can be used by Swahili news platforms to automatically categorize news articles, improving accessibility and navigation for readers.

## Dataset
The dataset consists of Swahili news articles categorized into five classes:
- **Kitaifa (National)**
- **Kimataifa (International)**
- **Biashara (Business)**
- **Michezo (Sports)**
- **Burudani (Entertainment)**

To run the project, you only need to provide the correct paths to the training and testing dataset files. The dataset preprocessing steps, including tokenization, padding, and encoding, are handled internally.

## Models Implemented
### 1. LSTM Model
- Embedding Layer
- LSTM Layer for capturing sequential dependencies
- Dense layer with softmax activation for classification

### 2. CNN Model
- Embedding Layer
- 1D Convolutional Layer for feature extraction
- Global Max Pooling for dimensionality reduction
- Dense layer with softmax activation

### 3. Transformer Model (Swahili BERT)
- Pretrained Swahili BERT model from Hugging Face
- Tokenization using BERT Tokenizer
- Fine-tuning using PyTorch on the Swahili dataset

## Installation & Requirements
This project does not require any external dependencies other than specifying the correct paths for the train and test dataset files. All necessary libraries are included within the Jupyter Notebook.

## Running the Project
To execute the code, simply run the Jupyter Notebook (`.ipynb`) file in the repository while ensuring the dataset paths are correctly set. The workflow includes:
1. Loading and preprocessing the dataset
2. Training LSTM and CNN models using TensorFlow/Keras
3. Fine-tuning the Swahili BERT model using PyTorch and Hugging Face
4. Evaluating model performance using Log Loss and accuracy
5. Making predictions on the test dataset

## References & Resources
### Hugging Face Model Used:
- [Swahili BERT for News Classification](https://huggingface.co/flax-community/bert-swahili-news-classification)

### Deep Learning Libraries:
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)

