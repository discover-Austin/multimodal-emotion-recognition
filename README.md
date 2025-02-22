# MultiModal Emotion Recognition
A deep learning system for real-time emotion recognition from both text and images using transformers.

## Features
- Real-time emotion detection from webcam feed
- Text sentiment analysis using BERT
- Fusion of multiple modalities for more accurate predictions
- Interactive web interface for demo
- Extensive documentation and tutorials
- Performance benchmarks and comparisons

## Project Overview
This repository implements a multimodal emotion recognition system that processes both images (from webcam feeds) and text (using BERT) to predict emotions. The system extracts features from images using a Vision Transformer and from text using BERT, then fuses these modalities to produce robust emotion predictions.

## Directory Structure
- **src/**  
  - **models/**: Contains model definition files.
  - **data/**
    - **preprocessing/**: Scripts for image and text preprocessing.
    - **augmentation/**: Image augmentation routines.
  - **training/**: Training scripts and checkpointing.
  - **utils/**: Utility files such as data loaders.
- **notebooks/**: Jupyter notebooks for exploration and testing.
- **docs/**: Extensive documentation and tutorials.
- **web_interface/**: FastAPI web interface to demonstrate real-time inference.
- **demo/**: Demo videos and related assets.
- **blog/**: Blog posts about development, benchmarks, and comparisons.
- **tests/**: Unit and integration tests for end-to-end verification.

## Setup and Installation
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/yourusername/multimodal-emotion-recognition.git
   cd multimodal-emotion-recognition