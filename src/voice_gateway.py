"""Core voice gateway service for Gabbie."""

import io
import logging
import queue
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import pyaudio
import requests
from openwakeword import get_pretrained_model_paths
from openwakeword.model import Model

from src.config import Config

logger = logging.getLogger(__name__)


class GatewayState(Enum):
    """Gateway states."""

    IDLE = "idle"
    LISTENING = "listening"
    DETECTED = "detected"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class GatewayEvent:
    """Event from the gateway."""

    event_type: str
    data: dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class VoiceGatewayService:
    """Voice gateway service that handles the complete voice interaction pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self._state = GatewayState.IDLE
        self._running = False
        self._event_queue: queue.Queue[GatewayEvent] = queue.Queue()
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._audio_thread: threading.Thread | None = None
        self._recording_thread: threading.Thread | None = None
        self._paudio: pyaudio.PyAudio | None = None
        self._stream = None
        self._wake_model: Model | None = None
        self._session = requests.Session()
        self._session.timeout = 30
        self._conversation_history: list[dict[str, Any]] = []
        self._last_confidence: float = 0.0
        self._wake_word_key: str = ""
        self._audio_level_frame_count: int = 0

    @property
    def state(self) -> GatewayState:
        """Get current gateway state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if gateway is running."""
        return self._running

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return self._conversation_history.copy()

    def start(self) -> None:
        """Start the voice gateway."""
        if self._running:
            return

        logger.info("Starting voice gateway...")
        self._running = True
        self._load_wake_word_model()
        self._start_audio_stream()
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._audio_thread.start()
        self._set_state(GatewayState.LISTENING)
        logger.info("Voice gateway started, listening for wake word")

    def stop(self) -> None:
        """Stop the voice gateway."""
        if not self._running:
            return

        logger.info("Stopping voice gateway...")
        self._running = False
        self._set_state(GatewayState.IDLE)

        if self._audio_thread:
            self._audio_thread.join(timeout=2)

        if self._recording_thread:
            self._recording_thread.join(timeout=2)

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass

        if self._paudio:
            try:
                self._paudio.terminate()
            except Exception:
                pass

        self._stream = None
        self._paudio = None
        logger.info("Voice gateway stopped")

    def _load_wake_word_model(self) -> None:
        """Load the wake word detection model."""
        logger.info(f"Loading wake word model: {self.config.wake_word}")

        model_path = None
        for path in get_pretrained_model_paths():
            if (
                self.config.wake_word in path
                or self.config.wake_word.replace("_", "-") in path
            ):
                model_path = path
                break

        if not model_path and self.config.wake_word == "hey_rhasspy":
            for path in get_pretrained_model_paths():
                if "alexa" in path:
                    model_path = path
                    logger.info("Using alexa as fallback")
                    break

        if not model_path:
            raise FileNotFoundError(
                f"No wake word model found for '{self.config.wake_word}'"
            )

        self._wake_model = Model(
            wakeword_model_paths=[model_path],
            enable_speex_noise_suppression=False,
            vad_threshold=self.config.vad_threshold,
        )
        # Store the actual key used in predictions (filename stem, e.g. "alexa_v0.1")
        self._wake_word_key = next(iter(self._wake_model.models.keys()))
        logger.info(f"Wake word model loaded, prediction key: {self._wake_word_key}")

    def _start_audio_stream(self) -> None:
        """Start the audio input stream."""
        self._paudio = pyaudio.PyAudio()

        device_info = None
        if self.config.input_device_index is not None:
            try:
                device_info = self._paudio.get_device_info_by_index(
                    self.config.input_device_index
                )
                logger.info(
                    f"Using input device: {device_info.get('name')} (index {self.config.input_device_index})"
                )
            except Exception as e:
                logger.error(
                    f"Invalid input device index {self.config.input_device_index}: {e}"
                )
                raise

        kwargs = {
            "format": pyaudio.paInt16,
            "channels": self.config.channels,
            "rate": self.config.sample_rate,
            "input": True,
            "frames_per_buffer": self.config.chunk_size,
            "stream_callback": self._audio_callback,
        }

        if self.config.input_device_index is not None:
            kwargs["input_device_index"] = self.config.input_device_index

        self._stream = self._paudio.open(**kwargs)
        self._stream.start_stream()
        logger.info("Audio stream started")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.debug(f"Audio stream status: {status}")
        self._audio_queue.put(in_data)
        self._audio_level_frame_count += 1
        if self._audio_level_frame_count >= 10:
            self._audio_level_frame_count = 0
            audio_array = np.frombuffer(in_data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)) / 32768.0)
            self._push_event("audio_level", {"level": rms})
        return (None, pyaudio.paContinue)

    def _audio_loop(self) -> None:
        """Main audio processing loop for wake word detection."""
        while self._running:
            if self._state != GatewayState.LISTENING:
                time.sleep(0.05)
                continue
            try:
                audio_data = self._audio_queue.get(timeout=0.1)
                if self._wake_model:
                    confidence = self._detect_wake_word(audio_data)
                    if logger.isEnabledFor(logging.DEBUG) and confidence > 0.1:
                        logger.debug(f"Wake word check: confidence = {confidence:.4f}")
                    if confidence:
                        self._wake_model.reset()
                        self._set_state(GatewayState.DETECTED)
                        self._push_event(
                            "wake_word_detected", {"confidence": self._last_confidence}
                        )
                        self._start_recording()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio loop: {e}")
                self._push_event("error", {"message": str(e)})

    def _detect_wake_word(self, audio_data: bytes) -> bool:
        """Detect wake word in audio data."""
        if not self._wake_model:
            return False

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        predictions = self._wake_model.predict(audio_array)

        if logger.isEnabledFor(logging.DEBUG):
            if predictions:
                for word, score in predictions.items():
                    if score > 0.01:
                        logger.debug(f"Wake word '{word}': {score:.4f}")

        if predictions and self._wake_word_key in predictions:
            score = predictions[self._wake_word_key]
            if score >= self.config.detection_threshold:
                self._last_confidence = float(score)
                return True
        return False

    def _start_recording(self) -> None:
        """Start recording user utterance after wake word."""
        self._set_state(GatewayState.RECORDING)
        self._recording_thread = threading.Thread(
            target=self._record_until_silence, daemon=True
        )
        self._recording_thread.start()

    def _record_until_silence(self) -> None:
        """Record audio until silence is detected or timeout."""
        frames = []
        silence_start = None
        start_time = time.time()

        # Discard audio that built up in the queue during wake word detection
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        while (
            self._running
            and time.time() - start_time < self.config.post_wake_buffer + 5
        ):
            try:
                data = self._audio_queue.get(timeout=0.1)
                frames.append(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                normalized_rms = rms / 32768.0

                if normalized_rms < self.config.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.config.silence_duration:
                        logger.info(
                            f"Silence detected after {time.time() - start_time:.1f}s"
                        )
                        break
                else:
                    silence_start = None

            except queue.Empty:
                continue

        audio_bytes = b"".join(frames)
        logger.info(
            f"Recorded {len(audio_bytes)} bytes ({len(audio_bytes) / 2 / self.config.sample_rate:.1f}s)"
        )

        if len(audio_bytes) >= 1000:
            self._process_utterance(audio_bytes)
        else:
            logger.warning("Audio too short, skipping")
            self._set_state(GatewayState.LISTENING)

    def _process_utterance(self, audio_data: bytes) -> None:
        """Process a complete utterance through STT -> LLM -> TTS."""
        self._set_state(GatewayState.PROCESSING)

        try:
            # STT
            text = self._transcribe(audio_data)
            if not text or text.startswith("["):
                logger.warning(f"Skipping transcription: '{text}'")
                self._set_state(GatewayState.LISTENING)
                return

            self._push_event("transcription", {"text": text})

            # LLM
            response_text = self._get_llm_response(text)
            if not response_text:
                logger.warning("Empty LLM response")
                self._set_state(GatewayState.LISTENING)
                return

            self._push_event("llm_response", {"text": response_text})

            # Add to conversation history
            self._conversation_history.append(
                {
                    "timestamp": time.time(),
                    "user": text,
                    "assistant": response_text,
                }
            )

            # TTS
            self._set_state(GatewayState.SPEAKING)
            response_audio = self._synthesize_speech(response_text)
            self._push_event("tts_generated", {"length": len(response_audio)})

            # Play audio
            self._play_audio(response_audio)
            self._push_event("playback_complete", {})

        except Exception as e:
            logger.error(f"Error processing utterance: {e}")
            self._push_event("error", {"message": str(e)})

        finally:
            # Discard audio captured during processing/playback before listening again
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._set_state(GatewayState.LISTENING)

    def _transcribe(self, audio_data: bytes) -> str:
        """Send audio to server for transcription."""
        url = f"{self.config.server_url}/audio/transcriptions"

        wav_buffer = io.BytesIO()
        wav_buffer.write(self._create_wav_header(len(audio_data)))
        wav_buffer.write(audio_data)
        wav_buffer.seek(0)

        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        data = {"model": self.config.stt_model}

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(url, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                text = result.get("text", "").strip()
                logger.info(f"STT: '{text}'")
                return text
            except requests.exceptions.RequestException as e:
                logger.warning(f"STT attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise

        return ""

    def _get_llm_response(self, text: str) -> str:
        """Get LLM response for text."""
        url = f"{self.config.server_url}/chat/completions"

        payload = {
            "model": self.config.llm_model,
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

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(f"LLM: '{content}'")
                return content
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise

        return ""

    def _synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech from text."""
        url = f"{self.config.server_url}/audio/speech"

        payload = {
            "model": self.config.tts_model,
            "input": text,
            "voice": self.config.tts_voice,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(url, json=payload)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "unknown")
                logger.info(f"TTS: {len(response.content)} bytes, content-type: {content_type}")
                return response.content
            except requests.exceptions.RequestException as e:
                logger.warning(f"TTS attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise

        return b""

    def _play_audio(self, audio_data: bytes) -> None:
        """Play audio through speakers via ffplay."""
        try:
            proc = subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
                input=audio_data,
                capture_output=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode().strip())
            logger.info(f"Playback complete ({len(audio_data)} bytes)")
        except FileNotFoundError:
            msg = "ffplay not found — install ffmpeg"
            logger.error(msg)
            self._push_event("error", {"message": msg})
        except Exception as e:
            msg = f"Playback failed: {e}"
            logger.error(msg)
            self._push_event("error", {"message": msg})

    def _create_wav_header(self, data_size: int) -> bytes:
        """Create WAV file header."""
        num_channels = self.config.channels
        sample_rate = self.config.sample_rate
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8

        header = struct.pack("<4sI4s", b"RIFF", 36 + data_size, b"WAVE")
        header += struct.pack(
            "<4sIHHIIHH",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        header += struct.pack("<4sI", b"data", data_size)

        return header

    def _set_state(self, state: GatewayState) -> None:
        """Set gateway state and push event."""
        old_state = self._state
        self._state = state
        if old_state != state:
            self._push_event("state_change", {"state": state.value})

    def _push_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Push event to event queue."""
        event = GatewayEvent(event_type=event_type, data=data)
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass

    def get_event(
        self, block: bool = True, timeout: float | None = None
    ) -> GatewayEvent | None:
        """Get next event from queue."""
        try:
            return self._event_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_status(self) -> dict[str, Any]:
        """Get current gateway status."""
        return {
            "state": self._state.value,
            "running": self._running,
            "config": {
                "wake_word": self.config.wake_word,
                "server_url": self.config.server_url,
                "input_device": self.config.input_device_index,
                "output_device": self.config.output_device_index,
            },
            "conversation_count": len(self._conversation_history),
        }

    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
        self.config.save()
