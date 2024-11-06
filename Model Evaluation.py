Model Evaluation
Objective Metrics
# Install evaluation packages
!pip install pystoi pesq

from pystoi import stoi
from pesq import pesq
from jiwer import wer

def evaluate_model(model, dataloader):
    total_stoi = 0
    total_pesq = 0
    total_wer = 0
    for batch in dataloader:
        # Generate predictions
        predictions = model(batch['input'])
        # Compute metrics
        total_stoi += stoi(batch['target'], predictions, sr=22050)
        total_pesq += pesq(22050, batch['target'], predictions, 'wb')
        total_wer += wer(batch['transcript'], batch['prediction_text'])
    # Average metrics
    avg_stoi = total_stoi / len(dataloader)
    avg_pesq = total_pesq / len(dataloader)
    avg_wer = total_wer / len(dataloader)
    print(f"STOI: {avg_stoi}, PESQ: {avg_pesq}, WER: {avg_wer}")

Subjective Listening Tests

# Use Gradio for setting up the interface
!pip install gradio

import gradio as gr

def tts_demo(text):
    # Generate audio
    audio = synthesize(text)
    return (22050, audio)

iface = gr.Interface(
    fn=tts_demo,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.outputs.Audio(type="numpy"),
    title="Gujarati TTS Demo"
)
iface.launch()


