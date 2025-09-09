#!/usr/bin/env python3
"""
Multimodal AI Chat Application
Combines webcam vision, voice input, and text chat with OpenAI GPT-4 Vision
"""

import cv2
import numpy as np
import base64
import io
import json
import time
import threading
import queue
import os
import sys
import select
from typing import Optional, Tuple
import speech_recognition as sr
import pygame
from pydub import AudioSegment
from pydub.playback import play
import openai
from dotenv import load_dotenv
from PIL import Image

# Try to import TTS libraries with coqui-tts priority
TTS_AVAILABLE = False
TTS_LIBRARY = None

try:
    # Try coqui-tts (modern package for Python 3.12)
    from TTS.api import TTS  # type: ignore
    TTS_AVAILABLE = True
    TTS_LIBRARY = "coqui-tts"
    print("✅ Using coqui-tts library")
except ImportError:
    try:
        # Fallback to pyttsx3 for basic TTS
        import pyttsx3  # type: ignore
        TTS_AVAILABLE = True
        TTS_LIBRARY = "pyttsx3"
        print("✅ Using pyttsx3 fallback")
    except ImportError:
        TTS_AVAILABLE = False
        TTS_LIBRARY = None
        print("⚠️ No TTS library available - speech output will be text-only")

# Load environment variables
load_dotenv()

class MultimodalChat:
    def __init__(self):
        """Initialize the multimodal chat system."""
        self.setup_openai()
        self.setup_tts()
        self.setup_speech_recognition()
        self.setup_camera()
        self.setup_audio()
        
        # Chat state
        self.conversation_history = []
        self.is_listening = False
        self.is_speaking = False
        
        print("🎥 Multimodal AI Chat initialized!")
        print("Commands:")
        print("  'v' - Voice input")
        print("  'c' - Capture image and chat")
        print("  'vc' - Voice input with camera capture")
        print("  'q' - Quit")
        print("  Type text and press Enter for text chat")
    
    def setup_openai(self):
        """Setup OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            print("Please set it in .env file or export it")
            sys.exit(1)
        
        self.client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
    
    def setup_tts(self):
        """Setup Text-to-Speech with better error handling."""
        if not TTS_AVAILABLE:
            print("⚠️ TTS not available - speech output disabled")
            self.tts = None
            self.tts_model_name = None
            return
            
        try:
            # Try different TTS models for better quality
            models_to_try = [
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ljspeech/fast_pitch"
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"🔄 Trying TTS model: {model_name}")
                    self.tts = TTS(model_name=model_name)
                    self.tts_model_name = model_name
                    print(f"✅ TTS initialized with model: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
                    continue
            else:
                print("❌ All TTS models failed, speech output disabled")
                self.tts = None
                self.tts_model_name = None
                
        except Exception as e:
            print(f"❌ TTS setup failed: {e}")
            self.tts = None
            self.tts_model_name = None
    
    def setup_audio(self):
        """Setup pygame for better audio playback."""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            print("✅ Audio system initialized")
        except Exception as e:
            print(f"⚠️ Audio system setup failed: {e}")
    
    def setup_speech_recognition(self):
        """Setup speech recognition."""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ Speech recognition initialized")
    
    def setup_camera(self):
        """Setup camera."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("❌ Could not open camera")
            print("Trying alternative camera indices...")
            for i in range(1, 5):
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    print(f"✅ Camera {i} opened successfully")
                    break
            else:
                print("❌ No camera found")
                self.camera = None
        else:
            print("✅ Camera initialized")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def encode_image(self, frame: np.ndarray) -> str:
        """Encode image to base64 for OpenAI API."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for efficiency
        height, width = rgb_frame.shape[:2]
        if width > 1024:
            scale = 1024 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return img_base64
    
    def listen_for_voice(self) -> Optional[str]:
        """Listen for voice input and return transcribed text."""
        if self.is_speaking:
            print("⏸️  Currently speaking, please wait...")
            return None
        
        print("🎤 Listening... (speak now)")
        self.is_listening = True
        
        try:
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=20)
            
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"📝 Heard: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand speech")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
        finally:
            self.is_listening = False
    
    def speak_text(self, text: str):
        """Convert text to speech and play it with improved audio handling."""
        if self.tts is None or self.is_speaking:
            print(f"🤖 {text}")
            return
        
        self.is_speaking = True
        print(f"🔊 Speaking: {text}")
        
        try:
            # Generate audio to a temporary file for better compatibility
            temp_audio_file = "temp_speech.wav"
            self.tts.tts_to_file(text=text, file_path=temp_audio_file)
            
            # Play audio in a separate thread using pygame
            def play_audio():
                try:
                    pygame.mixer.music.load(temp_audio_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_audio_file):
                        os.remove(temp_audio_file)
                        
                except Exception as e:
                    print(f"❌ Audio playback error: {e}")
                    print(f"🤖 {text}")
                finally:
                    self.is_speaking = False
            
            thread = threading.Thread(target=play_audio)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            print(f"🤖 {text}")
            self.is_speaking = False
    
    def chat_with_vision(self, image_data: str, text_query: str) -> str:
        """Send image and text to OpenAI GPT-4 Vision."""
        try:
            # Use conversation history for better context
            messages = []
            
            # Add recent conversation context (last 3 exchanges)
            if len(self.conversation_history) > 0:
                recent_history = self.conversation_history[-6:]  # Last 3 exchanges (user + assistant)
                for msg in recent_history:
                    if msg["role"] in ["user", "assistant"]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current query with image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            })
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def chat_text_only(self, text_query: str) -> str:
        """Send text-only query to OpenAI GPT-4."""
        try:
            # Use conversation history for better context
            messages = []
            
            # Add recent conversation context (last 5 exchanges)
            if len(self.conversation_history) > 0:
                recent_history = self.conversation_history[-10:]  # Last 5 exchanges
                messages.extend(recent_history)
            
            # Add current query
            messages.append({"role": "user", "content": text_query})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def process_voice_input(self):
        """Process voice input and respond."""
        text = self.listen_for_voice()
        if text:
            self.process_text_input(text)
    
    def process_voice_with_vision(self):
        """Process voice input with camera capture."""
        text = self.listen_for_voice()
        if text:
            self.process_vision_input(text)
    
    def process_text_input(self, text: str):
        """Process text input and respond."""
        if not text.strip():
            return
        
        print(f"👤 You: {text}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": text})
        
        # Get response
        response = self.chat_text_only(text)
        print(f"🤖 AI: {response}")
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Speak response
        self.speak_text(response)
    
    def process_vision_input(self, text_query: str):
        """Process vision input with captured image."""
        print("📷 Capturing image...")
        frame = self.capture_frame()
        
        if frame is None:
            print("❌ Could not capture image")
            return
        
        # Show captured image briefly
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow("Captured Image")
        
        # Encode image
        image_data = self.encode_image(frame)
        
        print(f"👤 You: {text_query}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": f"[Looking at camera] {text_query}"})
        
        # Get vision response
        response = self.chat_with_vision(image_data, text_query)
        print(f"🤖 AI: {response}")
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Speak response
        self.speak_text(response)
    
    def run(self):
        """Main application loop."""
        print("\n🎥 Multimodal AI Chat is running!")
        print("Press 'q' to quit, 'v' for voice, 'c' for camera, 's' for voice+camera")
        
        try:
            while True:
                # Show live camera feed
                if self.camera is not None:
                    ret, frame = self.camera.read()
                    if ret:
                        # Add instructions overlay
                        cv2.putText(frame, "v=voice | c=camera | s=voice+camera | q=quit", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Multimodal AI Chat", frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.process_voice_input()
                elif key == ord('c'):
                    print("\n🎤📷 Speak your question, then image will be captured...")
                    # Listen for voice input first
                    text_query = self.listen_for_voice()
                    if text_query and text_query.strip():
                        print("📷 Now capturing image...")
                        self.process_vision_input(text_query)
                    else:
                        print("❌ No voice input detected, please try again")
                elif key == ord('s'):  # New: voice + camera
                    print("\n🎤📷 Voice input with camera capture...")
                    self.process_voice_with_vision()
                
                # Check for text input (non-blocking)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    text_input = input("\n📝 You: ").strip()
                    if text_input:
                        self.process_text_input(text_input)
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Clean up any temporary audio files
        temp_files = ["temp_speech.wav"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Clean up pygame
        try:
            pygame.mixer.quit()
        except:
            pass
            
        print("🧹 Cleaned up resources")

def main():
    """Main entry point."""
    print("🚀 Starting Multimodal AI Chat...")
    
    # Check if running in interactive mode
    if not sys.stdin.isatty():
        print("❌ This application requires an interactive terminal")
        sys.exit(1)
    
    try:
        chat = MultimodalChat()
        chat.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()