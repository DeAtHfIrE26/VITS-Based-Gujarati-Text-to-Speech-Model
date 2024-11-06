# VITS-Based-Gujarati-Text-to-Speech-Model

Author: Patel Kashyap KalpeshkumarRegistration Number: 21BCE0216

Overview

This project implements a VITS (Variational Inference Text-to-Speech) model to generate high-quality audio from text in Gujarati. The model leverages deep learning to create natural-sounding synthesized speech, which can be useful for applications such as voice assistants, language learning tools, and personalized TTS systems.

Table of Contents

Overview

Project Features

Prerequisites

Setup Instructions

Training the Model

Generating Audio from Text

Evaluation

Troubleshooting

Contact

Project Features

Gujarati Text-to-Speech Conversion: Converts text written in Gujarati into natural-sounding audio.

Customizable Training: Allows training with a new dataset or language.

Checkpoint Saving: Saves the best model during training based on validation performance.

Interactive Visualization: Provides waveform and spectrogram visualizations of synthesized audio.

Prerequisites

Make sure you have the following installed:

Python 3.8 or newer

Google Colab or a machine with a CUDA-enabled GPU (for training)

PyTorch 1.9.0 or newer with GPU support

Required Python packages (listed in requirements.txt)

Setup Instructions

Clone the Repository

git clone https://github.com/your-repo-name/vits-gujarati-tts.git
cd vits-gujarati-tts

Install Dependencies
Create a virtual environment (optional but recommended) and install the necessary packages:

python3 -m venv vits_env
source vits_env/bin/activate  # On Windows use `vits_env\Scripts\activate`
pip install -r requirements.txt

Mount Google Drive (for Checkpoints)
To store checkpoints during training, mount your Google Drive in Google Colab:

from google.colab import drive
drive.mount('/content/drive')

Set Up Configuration

Modify the configs/config_gujarati.json file to match your dataset and language preferences.

Training the Model

Prepare Your Dataset

Ensure you have audio (.wav) and corresponding text (.txt) pairs ready for training.

Organize your dataset into training, validation, and test sets.

Run the Training Script

Train the model using the following command:

python train.py --config configs/config_gujarati.json

During training, model checkpoints will be saved to /content/drive/MyDrive/vits_checkpoints/.

Monitor Training

You can use print statements or TensorBoard to monitor the progress of training.

Generating Audio from Text

Load the Pre-trained Model

Ensure best_model.pth is available in /content/drive/MyDrive/vits_checkpoints/.

Run the Audio Generation Script

Use the script below to generate audio from input text:

import torch
import json
import numpy as np
import soundfile as sf
from IPython.display import Audio

# Load model and configuration
model_checkpoint = '/content/drive/MyDrive/vits_checkpoints/best_model.pth'
config_path = 'configs/config_gujarati.json'

with open(config_path, 'r') as f:
    config = json.load(f)

# Load model (replace with actual implementation)
model = torch.load(model_checkpoint, map_location='cpu')
model.eval()

# Input text
input_text = "તમારો ગુજરાતી ટેક્સ્ટ અહીં"
# Placeholder for text-to-sequence conversion and audio generation
# Replace with the actual implementation
synthesized_audio = np.sin(np.linspace(0, 1000, 22050))  # Example dummy audio

# Save and play synthesized audio
sf.write('synthesized_output.wav', synthesized_audio, config['sample_rate'])
Audio('synthesized_output.wav')

Evaluation

Run Evaluation Script: Use evaluate.py to assess the model's performance on the validation set.

Visualization: You can visualize the generated waveform and spectrogram using librosa and matplotlib:

import librosa
import librosa.display
import matplotlib.pyplot as plt

audio, sr = librosa.load('synthesized_output.wav', sr=None)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

Troubleshooting

FileNotFoundError: Ensure that all file paths are correct, and files are in the specified directories.

Checkpoint Not Found: Verify that best_model.pth is present in the checkpoints directory.

Training Issues: Double-check the dataset alignment and ensure GPU support is enabled if available.

Contact

If you have any questions or need support, please feel free to contact:

Patel Kashyap Kalpeshkumar

Registration Number: 21BCE0216

Email: your-email@example.com(Replace with your contact email)

Thank you for exploring this project! Feel free to contribute or reach out for collaboration.
