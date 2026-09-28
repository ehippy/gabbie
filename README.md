# gabbie

A hands-free voice assistant. Talk, it listens (voice-activity detection,
or hold down axon's physical button for explicit push-to-talk), transcribes
remotely, sends your words to an LLM, and speaks the reply back.

## Setup

```sh
uv run main.py
```

Run it in a real terminal with a working microphone — it needs a live TTY
and audio device, so it won't work piped through another tool.

Say "bye", "goodbye", "exit", or "quit" to end the conversation, or Ctrl+C.

## Architecture

- **Speech-to-text, LLM, and text-to-speech** all come from `neuralforge`,
  a local [Lemonade](https://lemonade-server.ai/) server exposing an
  OpenAI-compatible API at `http://neuralforge:13305/v1`, no API key
  needed. STT is Whisper-Tiny (`STT_MODEL` in `main.py`), sent a WAV clip
  via `/v1/audio/transcriptions`; the LLM is whatever's currently loaded
  there (see `DEFAULT_LLM_MODEL`); TTS is Kokoro (`kokoro-v1`). All three
  are GPU/NPU accelerated on that machine, which is why they're called out
  to instead of run locally — local CPU TTS (previously KittenTTS) was the
  dominant source of latency before that switch, and local CPU Whisper
  (`faster-whisper`) was the next one after that.
- The LLM's chain-of-thought ("thinking") is explicitly disabled per
  request (`enable_thinking: false`) since it roughly doubled reply latency
  for no benefit on casual conversation.
- **Push-to-talk**: axon's physical Whisplay button publishes `"start"`/
  `"stop"` over MQTT (`crumb:1883`, topic `axon/ptt` - see
  `axon/software/axon-native/src/mqtt.rs`'s `publish_ptt()`). Holding it
  bypasses VAD and the wake-word gate entirely - explicit hold is already
  the "I'm talking to you" signal. Say "Gabbie" (or a close-enough variant)
  first if you're not holding the button.
