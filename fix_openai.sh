#!/bin/bash

echo "🔧 Quick fix for OpenAI dependency issue..."

# Activate virtual environment
if [ -d "venv-multimodal" ]; then
    source venv-multimodal/bin/activate
    echo "✅ Activated virtual environment"
else
    echo "❌ Virtual environment not found. Please run setup_multimodal.sh first"
    exit 1
fi

# Force reinstall with compatible versions
echo "🔄 Force reinstalling compatible versions..."
pip uninstall -y openai httpx httpcore
pip install "httpcore==0.18.0"
pip install "httpx==0.24.1" 
pip install "openai==1.35.0"

echo "🧪 Testing fix..."
python3 -c "
try:
    import openai
    client = openai.OpenAI(api_key='test')
    print('✅ OpenAI client works!')
except Exception as e:
    print(f'❌ Still broken: {e}')
"

echo "✅ Fix complete! Try running your application again."
