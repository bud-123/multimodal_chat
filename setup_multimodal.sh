#!/bin/bash

echo "🔧 Setting up Multimodal AI Chat with fixed dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv-multimodal" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv-multimodal
fi

# Activate virtual environment
source venv-multimodal/bin/activate
echo "✅ Activated virtual environment"

# Upgrade pip and setuptools first
echo "⬆️ Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Completely clean up problematic packages
echo "🗑️ Removing ALL potentially conflicting packages..."
pip uninstall -y openai httpx httpcore h11 anyio sniffio certifi urllib3

# Install HTTP dependencies in specific order with exact compatible versions
echo "🔗 Installing HTTP stack with compatible versions..."
pip install "certifi>=2023.7.22"
pip install "sniffio>=1.3.0"
pip install "anyio>=3.7.1,<5.0.0"
pip install "h11>=0.14.0,<1.0.0"
pip install "httpcore>=0.18.0,<1.0.0"
pip install "httpx>=0.24.1,<0.26.0"

# Install compatible OpenAI version
echo "🤖 Installing compatible OpenAI client..."
pip install "openai>=1.3.0,<2.0.0"

# Install multimedia packages
echo "🎵 Installing multimedia packages..."
pip install pygame soundfile librosa pydub

# Install remaining requirements
echo "📋 Installing remaining requirements..."
pip install -r requirements-multimodal.txt --force-reinstall

# Test installations
echo "🧪 Testing critical imports..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test OpenAI
try:
    import openai
    print(f'✅ OpenAI import successful - version: {openai.__version__}')
    
    # Test client creation with compatible syntax
    client = openai.OpenAI(api_key='test-key')
    print('✅ OpenAI client creation works')
    
except Exception as e:
    print(f'❌ OpenAI error: {e}')
    sys.exit(1)

# Test other critical imports
try:
    import cv2
    print('✅ OpenCV import successful')
except ImportError:
    print('❌ OpenCV import failed')

try:
    import pygame
    print('✅ Pygame import successful')
except ImportError:
    print('❌ Pygame import failed')

try:
    from TTS.api import TTS
    print('✅ TTS import successful')
except ImportError as e:
    print(f'⚠️ TTS import failed: {e}')

try:
    import speech_recognition as sr
    print('✅ Speech Recognition import successful')
except ImportError:
    print('❌ Speech Recognition import failed')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "To run the application:"
    echo "1. Make sure your .env file contains your OPENAI_API_KEY"
    echo "2. Run: python multimodal_chat.py"
    echo ""
else
    echo ""
    echo "❌ Setup encountered errors. Please check the output above."
fi