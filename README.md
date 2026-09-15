# gabbie

A hands-free voice assistant. Talk, it listens (voice-activity detection, no
push-to-talk), transcribes locally, sends your words to an LLM, and speaks
the reply back.

## Setup

```sh
uv run main.py
```

Run it in a real terminal with a working microphone — it needs a live TTY
and audio device, so it won't work piped through another tool.

Say "bye", "goodbye", "exit", or "quit" to end the conversation, or Ctrl+C.

## Architecture

- **Speech-to-text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (`base.en`), local and CPU-only. Fast enough (well under a second per
  utterance) that there was no reason to send audio over the network for
  this part.
- **LLM + text-to-speech**: both come from `neuralforge`, a local
  [Lemonade](https://lemonade-server.ai/) server exposing an
  OpenAI-compatible API at `http://neuralforge:13305/v1`, no API key
  needed. The LLM is whatever's currently loaded there (see
  `LLM_MODEL` in `main.py`); TTS is Kokoro (`kokoro-v1`). Both are GPU/NPU
  accelerated on that machine, which is why they're called out to instead
  of run locally — local CPU TTS (previously KittenTTS) was the dominant
  source of latency before this switch.
- The LLM's chain-of-thought ("thinking") is explicitly disabled per
  request (`enable_thinking: false`) since it roughly doubled reply latency
  for no benefit on casual conversation.
