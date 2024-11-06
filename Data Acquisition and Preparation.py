Data Acquisition and Preparation

Mounting Google Drive

from google.colab import drive
drive.mount('/content/drive')


Data Download and Extraction
import os

# Set data paths
data_root = '/content/drive/MyDrive/svarah'
audio_dir = os.path.join(data_root, 'audio')
transcript_dir = os.path.join(data_root, 'transcripts')

# Verify data directories
if os.path.exists(audio_dir) and os.path.exists(transcript_dir):
    print("Data directories found.")
else:
    print("Data directories not found. Please check the paths.")
