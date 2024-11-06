Model Quantization and Pruning

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'quantized_model.pth')

Building a RESTful API with FastAPI

# main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch

app = FastAPI()

# Load the model
model = ...  # Load your trained model
model.eval()

@app.post("/synthesize")
async def synthesize_text(text: str):
    # Generate audio
    audio = synthesize(text)
    # Return audio file
    return StreamingResponse(io.BytesIO(audio), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


Dockerization for Production

FROM pytorch/pytorch:latest

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build the Docker image
!docker build -t gujarati-tts-app .

# Run the Docker container
!docker run -p 8000:8000 gujarati-tts-app


