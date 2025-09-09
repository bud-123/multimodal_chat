echo "🎥 Setting up Multimodal Webcam + Voice Chat..."

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Please set OPENAI_API_KEY environment variable"
    exit 1
fi

# Detect OS and install camera dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Installing Linux dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-opencv \
        libopencv-dev \
        v4l-utils \
        cheese \
        portaudio19-dev \
        ffmpeg \
        libsndfile1 \
        build-essential
        
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing macOS dependencies..."
    brew install opencv portaudio ffmpeg libsndfile
fi

# Create/activate virtual environment
if [ ! -d "venv-multimodal" ]; then
    python3 -m venv venv-multimodal
fi

source venv-multimodal/bin/activate

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip

# Install PyTorch (GPU if available)
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 Installing PyTorch with CUDA..."
    pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing PyTorch CPU..."
    pip install torch torchaudio
fi

# Install other dependencies
pip install \
    opencv-python \
    opencv-python-headless \
    pillow \
    numpy \
    openai \
    pyaudio \
    speechrecognition \
    pydub \
    TTS \
    requests \
    python-dotenv

# Test camera
echo "📷 Testing camera access..."
python3 -c "
import cv2
camera = cv2.VideoCapture(0)
if camera.isOpened():
    print('✅ Camera working!')
    camera.release()
else:
    print('❌ Camera not accessible')
    print('Try: sudo usermod -a -G video \$USER')
    print('Then logout and login again')
"

# Download TTS model
echo "📥 Pre-downloading TTS model..."
python3 -c "
from TTS.api import TTS
print('Downloading TTS model...')
tts = TTS(model_name='tts_models/en/ljspeech/glow-tts')
print('✅ TTS ready!')
"