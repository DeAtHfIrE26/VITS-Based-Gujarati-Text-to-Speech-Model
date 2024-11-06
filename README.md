# 🎙️ VITS-Based Gujarati Text-to-Speech Model

**Author**: Patel Kashyap Kalpeshkumar  
**Registration Number**: 21BCE0216  
**Email**: [kashyappatel2673@gmail.com](mailto:kashyappatel2673@gmail.com)  
**GitHub Repository**: [VITS-Based-Gujarati-Text-to-Speech-Model](https://github.com/DeAtHfIrE26/VITS-Based-Gujarati-Text-to-Speech-Model)

---

## 📌 Overview

This project provides a **VITS (Variational Inference Text-to-Speech)** model for synthesizing natural-sounding Gujarati speech from text. It’s ideal for applications like **voice assistants**, **language learning tools**, and **personalized TTS systems**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Setup Instructions](#-setup-instructions)
- [Training the Model](#-training-the-model)
- [Generating Audio](#-generating-audio)
- [Evaluation](#-evaluation)
- [Troubleshooting](#-troubleshooting)
- [Contact](#-contact)

---

## 🌟 Features

- **Gujarati TTS**: Converts Gujarati text into natural-sounding audio.
- **Customizable Training**: Supports retraining with new datasets or other languages.
- **Checkpoint Saving**: Automatically saves model checkpoints based on validation performance.
- **Visualization Tools**: Generates waveforms and spectrograms for synthesized audio.

---

## 🛠️ Prerequisites

Make sure you have:

- Python 3.8 or later
- CUDA-enabled GPU or Google Colab (recommended for training)
- PyTorch 1.9.0+ with GPU support
- All packages in `requirements.txt`

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/DeAtHfIrE26/VITS-Based-Gujarati-Text-to-Speech-Model.git
cd VITS-Based-Gujarati-Text-to-Speech-Model
2. Install Dependencies
Create a virtual environment (optional but recommended) and install required packages:
```
bash
Copy code
python3 -m venv vits_env
source vits_env/bin/activate  # On Windows, use `vits_env\Scripts\activate`
pip install -r requirements.txt
3. Mount Google Drive (for Colab Users)
To save checkpoints in Google Drive, use the following code in Colab:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
4. Edit Configuration
Modify configs/config_gujarati.json for dataset paths and language settings.

🏋️‍♂️ Training the Model
Dataset Preparation
Prepare paired audio (.wav) and text (.txt) files. Organize them into training, validation, and test folders.

Start Training
Run the training script with the following command:

bash
Copy code
python train.py --config configs/config_gujarati.json
Model checkpoints will be saved to /content/drive/MyDrive/vits_checkpoints/.

🎶 Generating Audio
Load Pre-trained Model
Ensure best_model.pth is in /content/drive/MyDrive/vits_checkpoints/.

Generate Audio from Text
Use the following script to generate audio from input text:

python
Copy code
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

# Load pre-trained model
model = torch.load(model_checkpoint, map_location='cpu')
model.eval()

# Input text
input_text = "તમારો ગુજરાતી ટેક્સ્ટ અહીં"

# Generate audio (dummy audio example, replace with actual TTS implementation)
synthesized_audio = np.sin(np.linspace(0, 1000, 22050))  # Replace with actual generation

# Save and play audio
sf.write('synthesized_output.wav', synthesized_audio, config['sample_rate'])
Audio('synthesized_output.wav')
📊 Evaluation
To evaluate the model on the validation set, use evaluate.py. You can visualize the waveform and spectrogram with the following code:

python
Copy code
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
🛠️ Troubleshooting
FileNotFoundError: Verify file paths and ensure all required files are correctly placed.
Checkpoint Missing: Confirm best_model.pth exists in the specified directory.
Training Errors: Check dataset formatting and enable GPU support if available.
📬 Contact
For any questions or collaboration requests, feel free to reach out!

Author: Patel Kashyap Kalpeshkumar
Registration Number: 21BCE0216
Email: kashyappatel2673@gmail.com
Thank you for exploring this project! Contributions and collaborations are welcome. 😊
