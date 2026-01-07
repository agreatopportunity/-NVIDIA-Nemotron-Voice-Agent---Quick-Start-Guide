## Features
```
🌧️ Matrix RainHyper blue/green animated matrix background
🔮 Pulsing OrbAnimated status indicator (ready/recording/processing)
💬 Chat UISmooth message bubbles with slide-in animations
📎 File UploadSupport for images, PDFs, docs, etc.
🎤 Voice ModeRecord and transcribe with visual feedback
📱 Mobile FriendlyResponsive design, touch optimized
⌨️ Text ModeType messages with optional TTS playback
```

# Copy the files
cp ~/Downloads/nemotron_web_ui.html .
cp ~/Downloads/nemotron_web_server.py .

# Install FastAPI if needed
pip install fastapi uvicorn python-multipart

# Run the server
python nemotron_web_server.py

# Open in browser
# http://localhost:8000


# Import and use
import NemotronVoiceUI from './components/NemotronVoiceUI';
API Endpoints (when using server)
EndpointMethodDescription/GETServe web UI/chatPOSTText chat/chat/speakPOSTChat + TTS audio/transcribePOSTUpload audio for ASR/synthesizePOSTText to speech/ws/voiceWebSocketReal-time voice/healthGETServer status
Mobile Experience
The UI is fully touch-optimized:

Large touch targets (44px+ buttons)
Swipe-friendly scrolling
No hover-dependent interactions
Viewport meta tag prevents zoom issues
Responsive font sizes with clamp()
