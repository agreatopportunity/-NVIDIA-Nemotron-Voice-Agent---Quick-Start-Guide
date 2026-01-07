#!/usr/bin/env python3
"""
NVIDIA Nemotron Voice Agent
===========================
A complete voice assistant using:
- ASR: Nemotron Speech Streaming 0.6B
- LLM: Nemotron Nano 9B (4-bit quantized)
- TTS: Silero TTS (fallback until Magpie is fully released)

Hardware: RTX 4060 Ti (16GB) + Titan V (12GB) + 64GB RAM
All models run on GPU 0 (4060 Ti) using ~10GB VRAM total

Usage:
    python nemotron_voice_agent.py           # Interactive mode selection
    python nemotron_voice_agent.py --text    # Start in text mode
    python nemotron_voice_agent.py --voice   # Start in voice mode
    python nemotron_voice_agent.py --think   # Enable reasoning mode

Controls:
    - Type 'voice' to switch to voice mode
    - Type 'text' to switch to text mode
    - In voice mode: Press ENTER to start/stop recording
    - Type 'clear' to clear conversation history
    - Type 'quit' to exit
"""

import os
import sys
import time
import wave
import tempfile
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VoiceAgentConfig:
    """Configuration for the voice agent."""
    # Device settings
    device: str = "cuda:0"  # 4060 Ti
    
    # ASR settings
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    sample_rate: int = 16000
    
    # LLM settings
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    use_reasoning: bool = False  # Set True for /think mode
    
    # TTS settings
    tts_sample_rate: int = 48000
    tts_speaker: str = "en_0"
    
    # System prompt
    system_prompt: str = """You are a helpful voice assistant running on NVIDIA Nemotron models. 
Keep your responses concise and conversational since they will be spoken aloud.
Be friendly and helpful."""

    # Conversation history
    max_history_turns: int = 10


# ============================================================================
# ASR Module (Nemotron Speech)
# ============================================================================

class NemotronASR:
    """Nemotron Speech ASR for transcription."""
    
    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self.model = None
        
    def load(self):
        """Load the ASR model."""
        print("📝 Loading Nemotron Speech ASR...")
        import nemo.collections.asr as nemo_asr
        
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.config.asr_model_name
        )
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ ASR loaded ({vram:.2f} GB VRAM)")
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if self.model is None:
            raise RuntimeError("ASR model not loaded")
            
        with torch.no_grad():
            transcriptions = self.model.transcribe([audio_path])
            
        return transcriptions[0] if transcriptions else ""


# ============================================================================
# LLM Module (Nemotron Nano 9B)
# ============================================================================

class NemotronLLM:
    """Nemotron Nano 9B LLM for response generation."""
    
    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.conversation_history: List[Dict[str, str]] = []
        
    def load(self):
        """Load the LLM with 4-bit quantization."""
        print("🧠 Loading Nemotron Nano 9B (4-bit)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            quantization_config=quantization_config,
            device_map={"": self.config.device},
            trust_remote_code=True
        )
        
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ LLM loaded ({vram:.2f} GB VRAM)")
        
    def generate(self, user_input: str) -> str:
        """Generate a response to user input."""
        if self.model is None:
            raise RuntimeError("LLM not loaded")
        
        # Build messages with history
        think_mode = "/think" if self.config.use_reasoning else "/no_think"
        messages = [
            {"role": "system", "content": f"{think_mode}\n{self.config.system_prompt}"}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history[-self.config.max_history_turns * 2:])
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.config.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.config.llm_max_tokens,
                do_sample=True,
                temperature=self.config.llm_temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("   🗑️  Conversation history cleared")


# ============================================================================
# TTS Module (Silero - fallback until Magpie is released)
# ============================================================================

class SileroTTS:
    """Silero TTS for speech synthesis."""
    
    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self.model = None
        
    def load(self):
        """Load the TTS model."""
        print("🔊 Loading Silero TTS...")
        
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en'
        )
        self.model = self.model.to(self.config.device)
        
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ TTS loaded ({vram:.2f} GB VRAM)")
        
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """Convert text to speech and save to file."""
        if self.model is None:
            raise RuntimeError("TTS model not loaded")
            
        # Generate audio
        audio = self.model.apply_tts(
            text=text,
            speaker=self.config.tts_speaker,
            sample_rate=self.config.tts_sample_rate
        )
        
        # Save to file
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
            
        # Convert to numpy and save
        audio_np = audio.cpu().numpy()
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.tts_sample_rate)
            wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            
        return output_path


# ============================================================================
# Audio I/O
# ============================================================================

class AudioRecorder:
    """Simple audio recorder using pyaudio."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self.recording = False
        self.stream = None
        self.audio = None
        
    def start_recording(self):
        """Start recording audio."""
        import pyaudio
        
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.recording = True
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._callback
        )
        
        self.stream.start_stream()
        
    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        if self.recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
        
    def stop_recording(self) -> str:
        """Stop recording and save to file."""
        self.recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.audio:
            self.audio.terminate()
            
        # Save to temp file
        output_path = tempfile.mktemp(suffix=".wav")
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(self.frames))
            
        return output_path


class AudioPlayer:
    """Simple audio player using pyaudio."""
    
    @staticmethod
    def play(audio_path: str):
        """Play an audio file."""
        import pyaudio
        
        with wave.open(audio_path, 'rb') as wav_file:
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=audio.get_format_from_width(wav_file.getsampwidth()),
                channels=wav_file.getnchannels(),
                rate=wav_file.getframerate(),
                output=True
            )
            
            chunk_size = 1024
            data = wav_file.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wav_file.readframes(chunk_size)
                
            stream.stop_stream()
            stream.close()
            audio.terminate()


# ============================================================================
# Voice Agent
# ============================================================================

class NemotronVoiceAgent:
    """Complete voice agent combining ASR, LLM, and TTS."""
    
    def __init__(self, config: Optional[VoiceAgentConfig] = None):
        self.config = config or VoiceAgentConfig()
        
        # Initialize components
        self.asr = NemotronASR(self.config)
        self.llm = NemotronLLM(self.config)
        self.tts = SileroTTS(self.config)
        
        self.recorder = AudioRecorder(sample_rate=self.config.sample_rate)
        self.player = AudioPlayer()
        
    def load_models(self):
        """Load all models."""
        print("\n" + "="*60)
        print("🚀 NVIDIA Nemotron Voice Agent")
        print("="*60)
        print(f"Device: {self.config.device}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        self.asr.load()
        self.llm.load()
        self.tts.load()
        
        total_vram = torch.cuda.memory_allocated(0) / 1024**3
        load_time = time.time() - start_time
        
        print(f"\n✅ All models loaded in {load_time:.1f}s")
        print(f"📊 Total VRAM used: {total_vram:.2f} GB")
        print("="*60 + "\n")
        
    def process_text(self, user_text: str) -> str:
        """Process text input and return text response."""
        return self.llm.generate(user_text)
    
    def process_audio(self, audio_path: str) -> tuple[str, str, str]:
        """
        Process audio input and return (transcript, response_text, response_audio_path).
        """
        # ASR: Audio -> Text
        print("   📝 Transcribing...")
        start = time.time()
        transcript = self.asr.transcribe(audio_path)
        asr_time = time.time() - start
        print(f"   ✓ ASR: {asr_time:.2f}s")
        
        # LLM: Generate response
        print("   🧠 Thinking...")
        start = time.time()
        response_text = self.llm.generate(transcript)
        llm_time = time.time() - start
        print(f"   ✓ LLM: {llm_time:.2f}s")
        
        # TTS: Text -> Audio
        print("   🔊 Synthesizing...")
        start = time.time()
        response_audio = self.tts.synthesize(response_text)
        tts_time = time.time() - start
        print(f"   ✓ TTS: {tts_time:.2f}s")
        
        return transcript, response_text, response_audio
    
    def show_menu(self) -> str:
        """Show startup menu and get user's mode choice."""
        print("\n🎤 Voice Agent Ready!")
        print("-" * 40)
        print("Select input mode:")
        print("  [1] Text mode  - Type your messages")
        print("  [2] Voice mode - Speak your messages")
        print("  [q] Quit")
        print("-" * 40)
        
        while True:
            choice = input("\nEnter choice (1/2/q): ").strip().lower()
            
            if choice in ['1', 'text', 't']:
                print("\n📝 Starting in TEXT mode")
                return "text"
            elif choice in ['2', 'voice', 'v']:
                print("\n🎤 Starting in VOICE mode")
                return "voice"
            elif choice in ['q', 'quit', 'exit']:
                return "quit"
            else:
                print("Invalid choice. Please enter 1, 2, or q.")
    
    def show_help(self):
        """Show available commands."""
        print("-" * 40)
        print("Commands:")
        print("  'text'    - Switch to text input mode")
        print("  'voice'   - Switch to voice input mode")
        print("  'clear'   - Clear conversation history")
        print("  'help'    - Show this help message")
        print("  'quit'    - Exit")
        print("-" * 40)
    
    def run_interactive(self, start_mode: Optional[str] = None):
        """Run interactive voice conversation loop."""
        
        # Determine starting mode
        if start_mode:
            mode = start_mode
            if mode == "text":
                print("\n📝 Starting in TEXT mode")
            elif mode == "voice":
                print("\n🎤 Starting in VOICE mode")
        else:
            mode = self.show_menu()
        
        if mode == "quit":
            print("\n👋 Goodbye!")
            return
        
        self.show_help()
        
        while True:
            try:
                if mode == "voice":
                    # Voice input mode
                    user_input = input("\n🎤 Press ENTER to record (or type command): ").strip().lower()
                    
                    # Check for commands first
                    if user_input == 'quit':
                        break
                    elif user_input == 'text':
                        mode = "text"
                        print("📝 Switched to text mode")
                        continue
                    elif user_input == 'clear':
                        self.llm.clear_history()
                        continue
                    elif user_input == 'help':
                        self.show_help()
                        continue
                    elif user_input == '':
                        # Empty input = start recording
                        print("🔴 Recording... (press ENTER to stop)")
                        
                        self.recorder.start_recording()
                        input()
                        audio_path = self.recorder.stop_recording()
                        print("⏹️  Recording stopped")
                        
                        # Process audio
                        transcript, response, audio_out = self.process_audio(audio_path)
                        
                        print(f"\n👤 You: {transcript}")
                        print(f"🤖 Assistant: {response}")
                        
                        # Play response
                        print("\n🔊 Playing response...")
                        self.player.play(audio_out)
                        
                        # Cleanup temp files
                        os.unlink(audio_path)
                        os.unlink(audio_out)
                    else:
                        # Treat as text input even in voice mode
                        print("   🧠 Thinking...")
                        response = self.llm.generate(user_input)
                        print(f"🤖 Assistant: {response}")
                        
                        speak = input("   🔊 Speak response? (y/N): ").strip().lower()
                        if speak == 'y':
                            audio_out = self.tts.synthesize(response)
                            self.player.play(audio_out)
                            os.unlink(audio_out)
                    
                else:  # text mode
                    user_input = input("\n👤 You: ").strip()
                    
                    if user_input.lower() == 'quit':
                        break
                    elif user_input.lower() == 'voice':
                        mode = "voice"
                        print("🎤 Switched to voice mode")
                        continue
                    elif user_input.lower() == 'clear':
                        self.llm.clear_history()
                        continue
                    elif user_input.lower() == 'help':
                        self.show_help()
                        continue
                    elif not user_input:
                        continue
                    
                    # Generate response
                    print("   🧠 Thinking...")
                    response = self.llm.generate(user_input)
                    print(f"🤖 Assistant: {response}")
                    
                    # Optionally speak response
                    speak = input("   🔊 Speak response? (y/N): ").strip().lower()
                    if speak == 'y':
                        audio_out = self.tts.synthesize(response)
                        self.player.play(audio_out)
                        os.unlink(audio_out)
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        print("\n👋 Goodbye!")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Check for pyaudio
    try:
        import pyaudio
    except ImportError:
        print("❌ pyaudio not installed. Install with:")
        print("   sudo apt install portaudio19-dev")
        print("   pip install pyaudio")
        sys.exit(1)
    
    # Create config
    config = VoiceAgentConfig()
    
    # Parse command line args
    start_mode = None
    
    if "--text" in sys.argv:
        start_mode = "text"
    elif "--voice" in sys.argv:
        start_mode = "voice"
        
    if "--think" in sys.argv:
        config.use_reasoning = True
        print("🧠 Reasoning mode enabled (/think)")
    
    # Create and run agent
    agent = NemotronVoiceAgent(config)
    agent.load_models()
    
    # Run interactive loop
    agent.run_interactive(start_mode=start_mode)


if __name__ == "__main__":
    main()
