import random

# Load noise files (ensure you have a set of noise audio files)
noise_files = glob.glob('/content/drive/MyDrive/noise_files/*.wav')

def add_background_noise(y, sr, noise_files, snr_db):
    noise_file = random.choice(noise_files)
    noise_y, _ = librosa.load(noise_file, sr=sr)
    # Trim or pad noise to match the length of y
    if len(noise_y) > len(y):
        noise_y = noise_y[:len(y)]
    else:
        noise_y = np.pad(noise_y, (0, len(y) - len(noise_y)), 'wrap')
    # Calculate signal power and adjust noise power accordingly
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise_y**2)
    desired_noise_power = signal_power / (10**(snr_db / 10))
    noise_y = noise_y * np.sqrt(desired_noise_power / noise_power)
    # Add noise to signal
    y_noisy = y + noise_y
    return y_noisy

# Create noisy audio files
noisy_data = []
for idx, row in tqdm(transcripts.iterrows(), total=transcripts.shape[0]):
    audio_path = os.path.join(audio_dir, row['filename'])
    y, sr = librosa.load(audio_path, sr=target_sr)
    # Apply noise injection
    y_noisy = add_background_noise(y, sr, noise_files, snr_db=20)
    # Save noisy audio
    noisy_filename = row['filename'].replace('.wav', '_noisy.wav')
    noisy_audio_path = os.path.join(audio_dir, noisy_filename)
    sf.write(noisy_audio_path, y_noisy, sr)
    # Add to transcripts
    noisy_data.append({
        'filename': noisy_filename,
        'text': row['text'],
        'normalized_text': row['normalized_text'],
        'phonemes': row['phonemes']
    })

# Append noisy data to transcripts
noisy_transcripts = pd.DataFrame(noisy_data)
transcripts = pd.concat([transcripts, noisy_transcripts], ignore_index=True)
