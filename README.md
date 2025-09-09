# 🎓 Multimodal Tutoring Pipeline - GPT-OSS Hackathon Prep

A **proof-of-concept** multimodal chat application built to test and validate the **voice + camera pipeline** for real-time AI tutoring. This project serves as the technical foundation for our GPT-OSS hackathon submission.

## 🎯 Project Purpose

**Scott Bernard** - An offline real-time linear algebra tutor with one eye.

*Named after X-Men's Cyclops and because we just like the name Bernard.*

**Mission**: Deploy an AI tutoring system that can:
- **See mathematical problems** through camera vision
- **Hear student questions** via voice interaction  
- **Provide real-time tutoring** for linear algebra
- **Work offline** using open-source models

**Hackathon Preparation** - Testing core components:
- **Voice interaction** for natural student-tutor conversations
- **Camera integration** for visual problem analysis (equations, matrices, graphs)
- **Real-time feedback** pipeline validation
- **Multimodal workflow** proof-of-concept

## ✅ Validated Components

- ✅ **Voice Input Pipeline**: Speech recognition with proper timeouts
- ✅ **Camera Capture**: Real-time image acquisition and processing  
- ✅ **Vision AI Integration**: GPT-4 Vision for visual analysis
- ✅ **Speech Output**: Text-to-speech response system
- ✅ **Multimodal Workflow**: Voice question → Camera capture → AI analysis

## 🚀 Quick Setup

### Prerequisites
- Python 3.12
- OpenAI API key (for current testing)
- Webcam + microphone

### Installation
```bash
# Clone and setup
cd multimodal_chat
python3.12 -m venv venv-multimodal
source venv-multimodal/bin/activate
pip install -r requirements-multimodal.txt

# Add API key (choose one option)
echo "OPENAI_API_KEY=your_key_here" > .env
# OR
export OPENAI_API_KEY=your_key_here

# Run test
python multimodal_chat_fixed.py
```

This gives users two convenient options:
1. **`.env` file** - Persistent across sessions
2. **export command** - Quick temporary setup for current session

The export option is especially useful for quick testing or when you don't want to create a file.

## 🎮 Testing the Pipeline

### Core Workflow (for tutoring validation)
1. **Press `c`** - Activates voice + camera mode
2. **Speak your question** - "What's wrong with this equation?" 
3. **Camera captures** - Automatically takes photo after voice input
4. **AI analyzes** - Both question and visual content
5. **Spoken response** - Natural tutoring feedback

### Controls
- `c` - **Voice + Camera** (primary tutoring workflow)
- `v` - Voice only
- `q` - Quit
- Type - Text chat

## 🔧 Technical Stack

**Core Pipeline Components:**
- **Speech**: Google Speech Recognition + Coqui TTS
- **Vision**: OpenCV + GPT-4 Vision (current) → **GPT-OSS** (hackathon)
- **Audio**: pygame mixer
- **AI**: OpenAI GPT-4 models (current) → **GPT-OSS** (hackathon)

**Key Files:**
- `multimodal_chat_fixed.py` - Main pipeline implementation
- `requirements-multimodal.txt` - Dependencies
- `.env` - API configuration

## 📊 Hackathon Readiness

### ✅ Proven Capabilities
- Natural voice interaction (10s timeout, 20s phrases)
- Reliable camera capture and encoding
- GPT-4 Vision integration for visual problem analysis
- Text-to-speech feedback with fallbacks
- Conversation context preservation

### 🚧 Known Limitations (for hackathon planning)
- Currently requires internet (OpenAI + Google APIs)
- API costs for extended testing
- Python 3.12 dependency for TTS
- ~2-3 second response latency

## 🎯 Hackathon Next Steps

### 🔄 Model Migration: OpenAI → GPT-OSS

**Architecture Change:**
```python
# Current: OpenAI API calls
self.client = openai.OpenAI(api_key=api_key)
response = self.client.chat.completions.create(model="gpt-4-turbo", ...)

# Hackathon: GPT-OSS integration
# Replace with local/remote GPT-OSS inference
```

### 🖥️ Deployment Options

#### Option A: GPT-OSS-120B (VM/Rented GPU)
```bash
# Recommended: Cloud VM with high-end GPU
# - NVIDIA A100/H100 for optimal performance
# - 80GB+ VRAM for 120B model
# - Deploy inference server on VM
# - Update client code to hit VM endpoint
```

**Deployment Stack:**
- **VM Provider**: GCP (primary), AWS/RunPod (alternatives)
- **Rented GPU**: Vast.ai for cost-effective GPU access
- **GPU Requirements**: A100 80GB or H100
- **Inference Server**: vLLM, TGI, or custom FastAPI
- **Client Updates**: Replace OpenAI calls with VM endpoint
- **Hardware Constraints**: Plug-in hardware (webcam, audio) may not be recognizable in VM environments

#### Option B: GPT-OSS-20B (Local Device)
```bash
# Local inference on M2 Mac Mini
# - Current setup: M2 chip with webcam
# - Quantized models (4-bit/8-bit) for efficiency
# - Direct local inference integration
```

**Local Setup:**
- **Hardware**: M2 Mac Mini (current setup) with external webcam
- **Memory**: Unified memory architecture advantage for large models
- **Quantization**: GPTQ/AWQ for memory efficiency
- **Framework**: Transformers + accelerate, or llama.cpp
- **Integration**: Direct model loading in Python
- **Advantage**: Full hardware compatibility (webcam, audio devices)

### 🔧 Code Migration Tasks

1. **Replace OpenAI client** with GPT-OSS inference calls
2. **Update image encoding** for GPT-OSS vision format
3. **Modify prompt templates** for GPT-OSS optimal performance
4. **Add local/remote inference switching** 
5. **Implement model loading** (local) or endpoint management (VM)

### 📈 Performance Targets for Hackathon

- **Response Latency**: <5 seconds for tutoring effectiveness
- **Model Accuracy**: Comparable educational assistance quality
- **Cost Efficiency**: Eliminate ongoing API costs
- **Offline Capability**: Reduce internet dependency (local deployment)

## 🏗️ Architecture Notes

**Current Pipeline:**
```
Voice Input → Speech Recognition → OpenAI API → TTS Output
Camera → Image Encoding → GPT-4 Vision → Response
```

**Hackathon Target:**
```
Voice Input → Speech Recognition → GPT-OSS (Local/VM) → TTS Output
Camera → Image Encoding → GPT-OSS Vision → Response
```

## 🐛 Common Issues & Fixes

```bash
# TTS not working
pip install --upgrade coqui-tts

# Camera permission issues  
# Check system camera permissions

# Import errors in IDE
# Set Python interpreter to venv-multimodal/bin/python
```

## 🎪 Hackathon Context

This project validates the **technical feasibility** of real-time multimodal tutoring. The pipeline successfully demonstrates:

- **Natural interaction** patterns for educational use cases
- **Visual problem recognition** capabilities  
- **Responsive feedback** delivery
- **Integration complexity** for production planning

**Current Status**: ✅ Pipeline validated with OpenAI GPT-4  
**Hackathon Goal**: 🔄 Migrate to GPT-OSS for open-source tutoring solution

Built as **technical validation** for GPT-OSS hackathon submission focusing on AI-powered education tools with open-source model deployment.

---

**Status**: ✅ Pipeline validated - Ready for GPT-OSS migration

*Next milestone: Complete model migration and deployment testing*
