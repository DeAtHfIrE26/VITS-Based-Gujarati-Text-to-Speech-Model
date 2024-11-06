# Install necessary packages
!pip install g2p_en  # Example for English; replace with Gujarati G2P if available

# Assuming we have a pre-trained G2P model for Gujarati
# Placeholder function for G2P conversion
def transformer_g2p(text):
    # Implement or integrate a pre-trained Transformer-based G2P model for Gujarati
    # For the purpose of this example, we'll use the normalized text as-is
    return text

# Apply G2P conversion
transcripts['phonemes'] = transcripts['normalized_text'].apply(transformer_g2p)

# Display sample
transcripts.head()
