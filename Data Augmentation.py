Data Augmentation
Speed Pertubation

from audiomentations import Compose, TimeStretch

# Define augmentation pipeline
augment = Compose([
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5)
])

# Create augmented audio files
augmented_data = []
for idx, row in tqdm(transcripts.iterrows(), total=transcripts.shape[0]):
    audio_path = os.path.join(audio_dir, row['filename'])
    y, sr = librosa.load(audio_path, sr=target_sr)
    # Apply augmentation
    y_aug = augment(samples=y, sample_rate=sr)
    # Save augmented audio
    aug_filename = row['filename'].replace('.wav', '_aug.wav')
    aug_audio_path = os.path.join(audio_dir, aug_filename)
    sf.write(aug_audio_path, y_aug, sr)
    # Add to transcripts
    augmented_data.append({
        'filename': aug_filename,
        'text': row['text'],
        'normalized_text': row['normalized_text'],
        'phonemes': row['phonemes']
    })

# Append augmented data to transcripts
augmented_transcripts = pd.DataFrame(augmented_data)
transcripts = pd.concat([transcripts, augmented_transcripts], ignore_index=True)
