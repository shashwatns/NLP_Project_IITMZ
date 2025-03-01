# Swahili News Classification using Deep Learning Models

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
### 1. LSTM & CNN Models (Implemented in a separate `.ipynb` file- NLP Project_LSTM_CNN.ipynb)
- Embedding Layer
- LSTM Layer for capturing sequential dependencies
- 1D Convolutional Layer for feature extraction
- Global Max Pooling for dimensionality reduction
- Dense layer with softmax activation

### 2. Transformer Model (Swahili BERT) (Implemented in a separate `.ipynb` file- NLP Project_Transformers.ipynb)
- Pretrained Swahili BERT model from Hugging Face
- Tokenization using BERT Tokenizer
- Fine-tuning using PyTorch on the Swahili dataset

## Installation & Requirements
This project does not require any external dependencies other than specifying the correct paths for the train and test dataset files. All necessary libraries are included within the Jupyter Notebooks.

## Running the Project
To execute the code, simply run the Jupyter Notebooks (`.ipynb`) files in the repository while ensuring the dataset paths are correctly set. The workflow includes:
1. Loading and preprocessing the dataset
2. Training LSTM and CNN models using TensorFlow/Keras (in one notebook)
3. Fine-tuning the Swahili BERT model using PyTorch and Hugging Face (in a separate notebook)
4. Evaluating model performance using Log Loss and accuracy
5. Making predictions on the test dataset

## References & Resources
### Hugging Face Model Used:
- [Swahili BERT for News Classification](https://huggingface.co/flax-community/bert-swahili-news-classification)

### Deep Learning Libraries:
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
