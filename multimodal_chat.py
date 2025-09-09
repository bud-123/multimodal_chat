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
from typing import Optional, Tuple
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from TTS.api import TTS
import openai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

class MultimodalChat:
    def __init__(self):
        """Initialize the multimodal chat system."""
        self.setup_openai()
        self.setup_tts()
        self.setup_speech_recognition()
        self.setup_camera()
        
        # Chat state
        self.conversation_history = []
        self.is_listening = False
        self.is_speaking = False
        
        print("🎥 Multimodal AI Chat initialized!")
        print("Commands:")
        print("  'v' - Voice input")
        print("  'c' - Capture image and chat")
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
        """Setup Text-to-Speech."""
        try:
            self.tts = TTS(model_name="tts_models/en/ljspeech/glow-tts")
            print("✅ TTS initialized")
        except Exception as e:
            print(f"❌ TTS setup failed: {e}")
            self.tts = None
    
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
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
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
        """Convert text to speech and play it."""
        if self.tts is None or self.is_speaking:
            print(f"🤖 {text}")
            return
        
        self.is_speaking = True
        print(f"🔊 Speaking: {text}")
        
        try:
            # Generate audio
            audio_data = self.tts.tts(text)
            
            # Convert to playable format
            audio_segment = AudioSegment(
                audio_data,
                frame_rate=22050,
                sample_width=2,
                channels=1
            )
            
            # Play audio in a separate thread
            def play_audio():
                play(audio_segment)
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
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
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
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def chat_text_only(self, text_query: str) -> str:
        """Send text-only query to OpenAI GPT-4."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": text_query}
                ],
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
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        # Encode image
        image_data = self.encode_image(frame)
        
        print(f"👤 You: {text_query}")
        
        # Get vision response
        response = self.chat_with_vision(image_data, text_query)
        print(f"🤖 AI: {response}")
        
        # Speak response
        self.speak_text(response)
    
    def run(self):
        """Main application loop."""
        print("\n🎥 Multimodal AI Chat is running!")
        print("Press 'q' to quit, 'v' for voice, 'c' for camera, or type text")
        
        try:
            while True:
                # Show live camera feed
                if self.camera is not None:
                    ret, frame = self.camera.read()
                    if ret:
                        # Add instructions overlay
                        cv2.putText(frame, "Press 'v' for voice, 'c' for camera, 'q' to quit", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Multimodal AI Chat", frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.process_voice_input()
                elif key == ord('c'):
                    text_query = input("\n📝 What would you like to know about what you see? ")
                    if text_query.strip():
                        self.process_vision_input(text_query)
                
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
    import select
    main()