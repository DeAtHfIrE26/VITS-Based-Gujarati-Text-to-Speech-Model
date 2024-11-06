# Create a directory for the dataset
!mkdir -p /content/data/svarah

# Download the Svarah dataset using the provided link
!wget -O /content/data/svarah/svarah.tar "https://indic-asr-public.objectstore.e2enetworks.net/svarah.tar"

# Extract the tar file
%cd /content/data/svarah
!tar -xvf svarah.tar

import glob
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Collect all audio files
audio_paths = glob.glob(os.path.join(audio_dir, '*.wav'))

# Set target sample rate
target_sr = 22050

# Function to preprocess audio
def preprocess_audio(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=True)
        # Resample if necessary
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        # Normalize audio
        y = y / np.max(np.abs(y))
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        # Save preprocessed audio
        sf.write(file_path, y, target_sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Remove corrupted file
        os.remove(file_path)

# Preprocess all audio files
for path in tqdm(audio_paths):
    preprocess_audio(path)
