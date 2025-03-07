<div align="center"> <img src="https://capsule-render.vercel.app/api?type=venom&height=300&color=gradient&customColorList=12&text=VITS-Based%20Gujarati%20TTS&animation=fadeIn&fontColor=fff&fontSize=60&fontAlignY=35&desc=Natural%20Speech%20Synthesis%20Using%20Variational%20Inference&descSize=20&descAlignY=60&stroke=fff&strokeWidth=2"> <div> <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&multiline=true&random=false&width=600&height=120&lines=Gujarati+Text-to-Speech;Variational+Inference+Model;Natural+Voice+Synthesis;Deep+Learning+Audio+Generation" alt="Typing SVG" /> </div> <p align="center"> </p> <div class="author-info" style="margin: 20px 0;"> <img src="https://img.shields.io/badge/Author-Patel_Kashyap_Kalpeshkumar-5046E5?style=for-the-badge&logo=github&logoColor=white" alt="Author"/> <img src="https://img.shields.io/badge/Registration-21BCE0216-5E60CE?style=for-the-badge&logo=academia&logoColor=white" alt="Registration"/> <a href="mailto:kashyappatel2673@gmail.com"> <img src="https://img.shields.io/badge/Email-kashyappatel2673@gmail.com-7B68EE?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/> </a> </div> <div class="tech-stack" style="margin: 15px 0;"> <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/> <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow"/> <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/> <img src="https://img.shields.io/badge/gujarati-FF4B4B?style=for-the-badge&logo=google-translate&logoColor=white" alt="Gujarati"/> </div> </div><div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>

## 🔊 Project Overview
<img align="right" width="350" >
This project implements a state-of-the-art VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) model specifically designed for the Gujarati language. By leveraging conditional variational autoencoder techniques with adversarial training, this system produces remarkably natural-sounding speech from Gujarati text input.

The model architecture combines the power of:

Flow-based decoders for high-fidelity audio generation

Stochastic duration predictors for natural rhythm and timing

Adversarial training for enhanced audio quality

This technology is particularly valuable for applications including voice assistants, accessibility tools, language learning platforms, and content localization systems targeting Gujarati-speaking populations.

<div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>

## ✨ Key Features
<table> <tr> <td width="50%"> <h3>🗣️ Gujarati Speech Synthesis</h3> <ul> <li>Converts Gujarati Unicode text to natural-sounding speech</li> <li>Supports proper pronunciation of Gujarati phonemes</li> <li>Maintains appropriate prosody and intonation</li> </ul> </td> <td width="50%"> <h3>🧠 Advanced Neural Architecture</h3> <ul> <li>Conditional variational autoencoder with flow-based decoder</li> <li>Stochastic duration predictor for natural timing</li> <li>Multi-period discriminator for adversarial training</li> </ul> </td> </tr> <tr> <td width="50%"> <h3>🔄 Flexible Training Pipeline</h3> <ul> <li>Custom dataset integration capabilities</li> <li>Distributed training support for faster convergence</li> <li>Automatic checkpoint management based on validation metrics</li> </ul> </td> <td width="50%"> <h3>📊 Comprehensive Visualization</h3> <ul> <li>Real-time training metrics monitoring</li> <li>Mel-spectrogram and waveform visualization</li> <li>Attention alignment visualization for debugging</li> </ul> </td> </tr> </table><div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>


## 🛠️ Prerequisites
Before setting up the project, ensure you have the following:

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;"> <div style="flex: 1; min-width: 300px; margin-right: 20px;"> <h3>💻 Hardware Requirements</h3> <ul> <li>CUDA-compatible GPU with at least 8GB VRAM</li> <li>16GB+ RAM for efficient data processing</li> <li>50GB+ storage space for datasets and checkpoints</li> </ul> </div> <div style="flex: 1; min-width: 300px;"> <h3>📚 Software Requirements</h3> <ul> <li>Python 3.8 or later</li> <li>PyTorch 1.9.0+ with CUDA support</li> <li>NVIDIA CUDA Toolkit 11.1+</li> <li>All dependencies listed in <code>requirements.txt</code></li> </ul> </div> </div><div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>
⚙️ Setup Instructions

---

## ⚙️ Setup Instructions
<div class="setup-container"> <div class="setup-step"> <h3>1. Clone the Repository</h3>
    
### 1. Clone the Repository
```bash
git clone https://github.com/DeAtHfIrE26/VITS-Based-Gujarati-Text-to-Speech-Model.git
cd VITS-Based-Gujarati-Text-to-Speech-Model
2. Install Dependencies
Create a virtual environment (optional but recommended) and install required packages:
```
</div> <div class="setup-step"> <h3>3. Install Dependencies</h3>

```bash
pip install -r requirements.txt
```
</div> <div class="setup-step"> <h3>4. Mount Google Drive (for Colab Users)</h3>

```python
from google.colab import drive
drive.mount('/content/drive')
```

</div> <div class="setup-step"> <h3>5. Configure the Model</h3>

``` bash
# Edit the configuration file to match your dataset paths and language settings
nano configs/config_gujarati.json
```

</div> </div> <div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>


## 🏋️‍♂️ Training the Model

<img align="right" width="300" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXg1OGl5c3NnMmRhNWZnNDdxcHJvbGNwYTZkMXVpMDVhZDRiYnJsYyZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/l46Cy1rHbQ92uuLXa/giphy.gif" alt="Training Animation">

Dataset Preparation
  1.Organize your dataset with paired audio (.wav) and text (.txt) files
  2.Structure folders into training, validation, and test directories
  3.Format text files with proper Gujarati Unicode characters

Training Process
```bash
# Start training with the Gujarati configuration
python train.py --config configs/config_gujarati.json
```

Training Parameters
<table> <tr> <th>Parameter</th> <th>Value</th> <th>Description</th> </tr> <tr> <td>Batch Size</td> <td>16</td> <td>Number of samples per training batch</td> </tr> <tr> <td>Learning Rate</td> <td>2e-4</td> <td>Initial learning rate for optimizer</td> </tr> <tr> <td>Epochs</td> <td>1000</td> <td>Maximum number of training epochs</td> </tr> <tr> <td>Checkpoint Interval</td> <td>5000</td> <td>Steps between model checkpoints</td> </tr> </table> <div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>



## 🎶 Generating Audio
<div class="code-container" style="background: linear-gradient(135deg, #5E60CE11, #FF5F6D11); padding: 20px; border-radius: 10px; border: 1px solid #5E60CE33;">

Load Pre-trained Model
```python
import torch
import json
import numpy as np
import soundfile as sf
from IPython.display import Audio
from model.vits import SynthesizerTrn

# Load model configuration
config_path = 'configs/config_gujarati.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Initialize model architecture
model = SynthesizerTrn(
    len(config["symbols"]),
    config["data"]["filter_length"] // 2 + 1,
    config["train"]["segment_size"] // config["data"]["hop_length"],
    **config["model"])

# Load pre-trained weights
model_checkpoint = '/content/drive/MyDrive/vits_checkpoints/best_model.pth'
model.load_state_dict(torch.load(model_checkpoint, map_location='cpu')['model'])
model.eval()
```
Generate Audio from Text

```python
# Gujarati text input
input_text = "તમારો ગુજરાતી ટેક્સ્ટ અહીં લખો"

# Text preprocessing
text_normalized = text_to_sequence(input_text, config["symbols"])
text_tensor = torch.LongTensor(text_normalized).unsqueeze(0)

# Generate audio
with torch.no_grad():
    audio = model.infer(text_tensor, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0]
    waveform = audio.squeeze().cpu().numpy()

# Save and play audio
sf.write('synthesized_output.wav', waveform, config['data']['sampling_rate'])
Audio('synthesized_output.wav')
``` 

</div> <div align="center"> <img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%"> </div>


## 📊 Evaluation
<div class="evaluation-container" style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;"> <div class="evaluation-card" style="flex: 1; min-width: 300px; background: linear-gradient(135deg, #5E60CE11, #FF5F6D11); padding: 20px; border-radius: 10px; border: 1px solid #5E60CE33;"> <h3>Objective Metrics</h3>


``` python
import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi

# Load reference and synthesized audio
ref_audio, sr = librosa.load('reference.wav', sr=None)
syn_audio, sr = librosa.load('synthesized_output.wav', sr=None)

# Calculate PESQ (Perceptual Evaluation of Speech Quality)
pesq_score = pesq(sr, ref_audio, syn_audio, 'wb')
print(f"PESQ Score: {pesq_score}")

# Calculate STOI (Short-Time Objective Intelligibility)
stoi_score = stoi(ref_audio, syn_audio, sr, extended=False)
print(f"STOI Score: {stoi_score}")
```
</div> <div class="evaluation-card" style="flex: 1; min-width: 300px; background: linear-gradient(135deg, #5E60CE11, #FF5F6D11); padding: 20px; border-radius: 10px; border: 1px solid #5E60CE33;"> <h3>Visualization Tools
    
## 🛠️ Troubleshooting
FileNotFoundError: Verify file paths and ensure all required files are correctly placed.
Checkpoint Missing: Confirm best_model.pth exists in the specified directory.
Training Errors: Check dataset formatting and enable GPU support if available.

## 📬 Contact
For any questions or collaboration requests, feel free to reach out!

Author: Patel Kashyap Kalpeshkumar
Registration Number: 21BCE0216
Email: kashyappatel2673@gmail.com
Thank you for exploring this project! Contributions and collaborations are welcome. 😊
