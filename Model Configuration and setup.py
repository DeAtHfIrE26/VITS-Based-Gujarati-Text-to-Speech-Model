Model Configuration and setup
Cloning Advanced VITS Repository
%cd /content
!git clone https://github.com/jaywalnut310/vits
%cd vits

**Multi-Speaker Support--
  # Add 'speaker_id' column to transcripts
# For simplicity, we'll assign a default speaker ID
transcripts['speaker_id'] = 0  # Replace with actual speaker IDs if available

# Update the filelists to include speaker IDs
def save_filelist_with_speaker(df, filename):
    with open(f'/content/filelists/{filename}', 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            audio_path = os.path.join(audio_dir, row['filename'])
            text = row['phonemes']
            speaker_id = row['speaker_id']
            f.write(f"{audio_path}|{speaker_id}|{text}\n")

# Split data and save filelists
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(transcripts, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

save_filelist_with_speaker(train_df, 'gujarati_train_filelist.txt')
save_filelist_with_speaker(val_df, 'gujarati_val_filelist.txt')
save_filelist_with_speaker(test_df, 'gujarati_test_filelist.txt')

