# Gabbie - Voice Gateway for Raspberry Pi 3

A lightweight voice assistant gateway optimized for resource-constrained devices (Raspberry Pi 3 with 512MB RAM).

## Architecture

```
┌─────────────────────────────────┐
│   Raspberry Pi 3 (512MB RAM)    │
│  ┌───────────────────────────┐  │
│  │ openwakeword (alexa)      │  │
│  │ ~50-100MB RAM usage       │  │
│  └───────────┬───────────────┘  │
│              │ wake detected    │
│  ┌───────────▼───────────────┐  │
│  │ 1. Record audio (16kHz)   │  │
│  │ 2. POST to /audio/transcriptions  │
│  │ 3. POST text to /chat/completions │
│  │ 4. POST text to /audio/speech     │
│  │ 5. Play returned audio    │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
         HTTP POST requests
```

## Requirements

### Hardware
- Raspberry Pi 3 (or similar, 512MB+ RAM)
- USB microphone or GPIO audio HAT
- Speakers or audio output

### Software (Pi)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio

# Install Python dependencies
uv sync
```

### Server
- Lemonade server or any OpenAI-compatible API server
- Must support:
  - STT: `/api/v1/audio/transcriptions` (Whisper)
  - LLM: `/api/v1/chat/completions`
  - TTS: `/api/v1/audio/speech` (Kokoro)

## Configuration

Edit the `Config` class in `main.py`:

```python
@dataclass
class Config:
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1280  # 80ms @ 16kHz
    
    # Wake word
    wake_word: str = "alexa"
    detection_threshold: float = 0.5
    vad_threshold: float = 0.5
    
    # Recording
    post_wake_buffer: int = 3  # seconds after wake word
    silence_threshold: float = 0.01
    silence_duration: float = 0.5  # end recording after this much silence
    
    # Server
    server_url: str = "http://neuralforge:8000/api/v1"
    tts_model: str = "kokoro-v1"
    tts_voice: str = "af_bella"
    stt_model: str = "Whisper-Tiny"
    llm_model: str = "Qwen3.5-122B-A10B-GGUF"
```

## Usage

```bash
# Run the voice gateway
python main.py
```

The application will:
1. Continuously listen for the wake word ("alexa")
2. When detected, record your utterance until silence
3. Send to server for processing (STT → LLM → TTS)
4. Play the response
5. Return to listening mode

## Available Wake Word Models

Check available models:
```bash
python -c "from openwakeword import get_pretrained_model_paths; [print(p.split('/')[-1]) for p in get_pretrained_model_paths()]"
```

Common options:
- `alexa` - "alexa"
- `hey_mycroft` - "hey mycroft"
- `hey_jarvis` - "hey jarvis"

## Performance

On Raspberry Pi 3 (512MB RAM):
- **openwakeword**: ~50-100MB RAM, minimal CPU
- **Audio capture**: ~10MB RAM
- **Network**: ~5MB RAM
- **Total**: ~150MB RAM (leaves room for OS)

## Troubleshooting

### No audio devices found
```bash
sudo apt-get install -y portaudio19-dev
```

### Wake word not detecting
- Increase `detection_threshold` (try 0.3-0.7)
- Enable `vad_threshold` for reduced false positives
- Check microphone levels: `arecord -l`

### Server connection errors
- Verify server is reachable: `ping neuralforge`
- Check API endpoints are available
- Adjust `max_retries` in config
