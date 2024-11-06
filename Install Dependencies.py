# Upgrade pip
!pip install --upgrade pip

# Install PyTorch with CUDA support
!pip install torch==1.13.1+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install essential libraries
!pip install numpy scipy matplotlib librosa unidecode inflect pandas tqdm
!pip install tensorboardX soundfile jieba

# Install advanced NLP libraries
!pip install transformers sentencepiece

# Install Phonemizer and its dependencies
!pip install phonemizer
!apt-get install espeak-ng -y

# Install apex for mixed-precision training
!git clone https://github.com/NVIDIA/apex.git
%cd apex
!pip install -v --disable-pip-version-check --no-cache-dir ./
%cd ..

# Install other necessary packages
!pip install tensorboard matplotlib seaborn

# Install FastAPI and Uvicorn for deployment
!pip install fastapi uvicorn

# Install Docker for deployment (if needed)
!apt-get install docker.io -y
