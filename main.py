#!/usr/bin/env python3
"""
Gabbie - Voice Gateway for Raspberry Pi 3

Continuously monitors for wake word ("alexa"), then:
1. Records user utterance
2. Sends to server for STT → LLM → TTS
3. Plays response audio
4. Returns to listening state

Optimized for low memory (512MB) and CPU constraints.
"""

import asyncio
import io
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openwakeword
import pyaudio
import requests
from openwakeword.model import Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class Config:
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1280  # 80ms @ 16kHz = 1280 bytes
    format: int = pyaudio.paInt16

    # Wake word settings
    wake_word: str = "alexa"
    detection_threshold: float = 0.5
    vad_threshold: float = 0.5

    # Recording settings
    post_wake_buffer: int = 3  # seconds to record after wake word
    silence_threshold: float = 0.01
    silence_duration: float = 0.5  # seconds of silence to end recording

    # Server settings
    server_url: str = "http://neuralforge:8000/api/v1"
    tts_model: str = "kokoro-v1"
    tts_voice: str = "af_bella"
    stt_model: str = "Whisper-Tiny"
    llm_model: str = "Qwen3.5-122B-A10B-GGUF"

    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0


config = Config()


class AudioRecorder:
    """Records audio from microphone at 16kHz."""

    def __init__(self):
        self.paudio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()

    def start(self):
        """Start recording from microphone."""
        try:
            self.stream = self.paudio.open(
                format=config.format,
                channels=config.channels,
                rate=config.sample_rate,
                input=True,
                frames_per_buffer=config.chunk_size,
                stream_callback=self._audio_callback,
            )
            self.is_recording = True
            self.stream.start_stream()
            logger.info("Audio recording started")
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            raise

    def stop(self):
        """Stop recording."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.is_recording = False
        self.paudio.terminate()
        logger.info("Audio recording stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.debug(f"Audio stream status: {status}")
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def record_until_silence(self, timeout: int = 10) -> bytes:
        """Record audio until silence is detected or timeout."""
        frames = []
        silence_start = None
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                data = self.audio_queue.get(timeout=0.1)
                frames.append(data)

                # Check for silence
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                normalized_rms = rms / 32768.0

                if normalized_rms < config.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= config.silence_duration:
                        logger.info(
                            f"Silence detected after {time.time() - start_time:.1f}s"
                        )
                        break
                else:
                    silence_start = None

            except queue.Empty:
                continue

        return b"".join(frames)


class WakeWordDetector:
    """Detects wake word using openwakeword."""

    def __init__(self):
        logger.info(f"Loading openWakeWord model: {config.wake_word}")

        # Get the model path for the wake word
        from openwakeword import get_pretrained_model_paths

        model_path = None
        for path in get_pretrained_model_paths():
            if config.wake_word in path or config.wake_word.replace("_", "-") in path:
                model_path = path
                break

        # Fallback: if hey_rhasspy not found, check for rhasspy variants
        if not model_path and config.wake_word == "hey_rhasspy":
            logger.info("hey_rhasspy not found, checking for available models...")
            for path in get_pretrained_model_paths():
                logger.info(f"  Available: {path.split('/')[-1]}")
            # Use alexa as fallback if hey_rhasspy not available
            for path in get_pretrained_model_paths():
                if "alexa" in path:
                    model_path = path
                    logger.info("Using alexa as fallback")
                    break

        if not model_path:
            raise FileNotFoundError(
                f"No wake word model found for '{config.wake_word}'"
            )

        try:
            self.model = Model(
                wakeword_model_paths=[model_path],
                enable_speex_noise_suppression=False,
                vad_threshold=config.vad_threshold,
            )
            logger.info("Wake word model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            raise

        self.is_detected = False
        self.detection_callback = None

    def process_audio(self, audio_data: bytes) -> bool:
        """Process audio frame and check for wake word."""
        # Convert bytes to numpy array for openwakeword
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        predictions = self.model.predict(audio_array)

        if predictions and config.wake_word in predictions:
            score = predictions[config.wake_word]
            if score >= config.detection_threshold:
                logger.info(f"Wake word detected! Confidence: {score:.3f}")
                return True
        return False

    def wait_for_wake_word(self) -> bool:
        """Block until wake word is detected."""
        recorder = AudioRecorder()
        recorder.start()

        try:
            while True:
                try:
                    audio_data = recorder.audio_queue.get(timeout=0.1)
                    if self.process_audio(audio_data):
                        return True
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            return False
        finally:
            recorder.stop()


class ServerClient:
    """Handles communication with the Lemonade server."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = requests.Session()
        self.session.timeout = 30

    def transcribe(self, audio_data: bytes) -> str:
        """Send audio to server for transcription."""
        url = f"{self.server_url}/audio/transcriptions"

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        wav_buffer.write(self._create_wav_header(len(audio_data)))
        wav_buffer.write(audio_data)
        wav_buffer.seek(0)

        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        data = {"model": config.stt_model}

        for attempt in range(config.max_retries):
            try:
                response = self.session.post(url, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                text = result.get("text", "").strip()
                logger.info(f"STT result: '{text}'")
                return text
            except requests.exceptions.RequestException as e:
                logger.warning(f"STT attempt {attempt + 1} failed: {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    raise

    def chat_completion(self, text: str) -> str:
        """Send text to LLM for response."""
        url = f"{self.server_url}/chat/completions"

        payload = {
            "model": config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful, concise voice assistant. Keep responses brief and natural for voice interaction.",
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        for attempt in range(config.max_retries):
            try:
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(f"LLM response: '{content}'")
                return content
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    raise

    def synthesize_speech(self, text: str) -> bytes:
        """Send text to TTS and get audio back."""
        url = f"{self.server_url}/audio/speech"

        payload = {
            "model": config.tts_model,
            "input": text,
            "voice": config.tts_voice,
        }

        for attempt in range(config.max_retries):
            try:
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                logger.info(f"TTS generated {len(response.content)} bytes of audio")
                return response.content
            except requests.exceptions.RequestException as e:
                logger.warning(f"TTS attempt {attempt + 1} failed: {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    raise

    def _create_wav_header(self, data_size: int) -> bytes:
        """Create WAV file header for 16kHz mono PCM."""
        import struct

        num_channels = config.channels
        sample_rate = config.sample_rate
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        subtitle_size = 16

        header = struct.pack("<4sI4s", b"RIFF", 36 + data_size, b"WAVE")
        header += struct.pack(
            "<4sIHHIIHH",
            b"fmt ",
            subtitle_size,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        header += struct.pack("<4sI", b"data", data_size)

        return header


class AudioPlayer:
    """Plays audio through speakers."""

    def __init__(self):
        self.paudio = pyaudio.PyAudio()
        self.stream = None

    def play(self, audio_data: bytes):
        """Play audio data."""
        try:
            self.stream = self.paudio.open(
                format=pyaudio.paInt16,
                channels=config.channels,
                rate=config.sample_rate,
                output=True,
            )
            self.stream.write(audio_data)
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Audio playback complete")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
        finally:
            self.paudio.terminate()


class Gabbie:
    """Main voice gateway application."""

    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.server = ServerClient(config.server_url)
        self.player = AudioPlayer()
        self.running = True

    def run(self):
        """Main loop: wait for wake word, then process request."""
        logger.info("=" * 50)
        logger.info("Gabbie Voice Gateway starting...")
        logger.info(f"Wake word: {config.wake_word}")
        logger.info(f"Server: {config.server_url}")
        logger.info("=" * 50)

        while self.running:
            try:
                # Step 1: Wait for wake word
                logger.info("Listening for wake word...")
                if not self._wait_for_wake():
                    break

                # Step 2: Record user utterance
                logger.info("Recording...")
                recorder = AudioRecorder()
                recorder.start()
                time.sleep(0.2)  # Small buffer after wake
                audio_data = recorder.record_until_silence(
                    timeout=config.post_wake_buffer + 5
                )
                recorder.stop()

                if len(audio_data) < 1000:
                    logger.warning("Audio too short, skipping")
                    continue

                logger.info(
                    f"Recorded {len(audio_data)} bytes ({len(audio_data) / config.sample_rate:.1f}s)"
                )

                # Step 3: Process through server pipeline
                try:
                    # STT
                    text = self.server.transcribe(audio_data)
                    if not text:
                        logger.warning("Empty transcription, skipping")
                        continue

                    # LLM
                    response_text = self.server.chat_completion(text)
                    if not response_text:
                        logger.warning("Empty LLM response, skipping")
                        continue

                    # TTS
                    response_audio = self.server.synthesize_speech(response_text)

                    # Step 4: Play response
                    logger.info(f"Playing response: '{response_text}'")
                    self.player.play(response_audio)

                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    continue

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                self.running = False
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(1)
                continue

    def _wait_for_wake(self) -> bool:
        """Wait for wake word detection."""
        recorder = AudioRecorder()
        recorder.start()

        try:
            while self.running:
                try:
                    audio_data = recorder.audio_queue.get(timeout=0.1)
                    if self.wake_detector.process_audio(audio_data):
                        return True
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            return False
        finally:
            recorder.stop()

        return False

    def stop(self):
        """Stop the application."""
        self.running = False


def main():
    """Entry point."""
    try:
        gabbie = Gabbie()
        gabbie.run()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
