Custom Text Processing
Text Normalization

!pip install indic-nlp-library
!git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git /content/indic_nlp_resources

# Set up Indic NLP Library
INDIC_RESOURCES_PATH = "/content/indic_nlp_resources"
from indicnlp import common
common.set_resources_path(INDIC_RESOURCES_PATH)
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Initialize the normalizer for Gujarati
normalizer_factory = IndicNormalizerFactory()
normalizer = normalizer_factory.get_normalizer("gu", remove_nuktas=False)

def advanced_normalize_text(text):
    # Unicode normalization
    text = normalizer.normalize(text)
    # Replace numerals with words (e.g., "3" -> "ત્રણ")
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
    text = ''.join([char if not char.isdigit() else UnicodeIndicTransliterator.transliterate(str(char), "hi", "gu") for char in text])
    # Remove unwanted characters
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

# Load transcripts
import pandas as pd
transcript_files = glob.glob(os.path.join(transcript_dir, '*.txt'))

# Create DataFrame
data = []
for transcript_file in tqdm(transcript_files):
    with open(transcript_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    filename = os.path.basename(transcript_file).replace('.txt', '.wav')
    data.append({'filename': filename, 'text': text})

transcripts = pd.DataFrame(data)

# Apply normalization
transcripts['normalized_text'] = transcripts['text'].apply(advanced_normalize_text)

# Display sample
transcripts.head()
