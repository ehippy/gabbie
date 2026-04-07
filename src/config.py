"""Configuration management for Gabbie."""

import os
import tomli
import tomli_w
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """Gabbie configuration."""

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1280
    input_device_index: int | None = None
    output_device_index: int | None = None

    # Wake word settings
    wake_word: str = "alexa"
    detection_threshold: float = 0.5
    vad_threshold: float = 0.5

    # Recording settings
    post_wake_buffer: int = 3
    silence_threshold: float = 0.01
    silence_duration: float = 0.5

    # Server settings
    server_url: str = "http://neuralforge:8000/api/v1"
    llm_url: str = "http://neuralforge:8000/v1"
    tts_model: str = "kokoro-v1"
    tts_voice: str = "af_bella"
    stt_model: str = "Whisper-Tiny"
    llm_model: str = "Qwen3.5-122B-A10B-GGUF"

    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Daemon settings
    daemon_socket_path: str = "~/.gabbie/gabbie.sock"
    pid_file: str = "~/.gabbie/gabbie.pid"

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        """Load configuration from TOML file."""
        if config_path is None:
            config_path = os.path.expanduser("~/.gabbie/config.toml")

        path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls(
            sample_rate=data.get("audio", {}).get("sample_rate", 16000),
            channels=data.get("audio", {}).get("channels", 1),
            chunk_size=data.get("audio", {}).get("chunk_size", 1280),
            input_device_index=data.get("audio", {}).get("input_device_index"),
            output_device_index=data.get("audio", {}).get("output_device_index"),
            wake_word=data.get("wake_word", {}).get("model", "alexa"),
            detection_threshold=data.get("wake_word", {}).get("threshold", 0.5),
            vad_threshold=data.get("wake_word", {}).get("vad_threshold", 0.5),
            post_wake_buffer=data.get("recording", {}).get("post_wake_buffer", 3),
            silence_threshold=data.get("recording", {}).get("silence_threshold", 0.01),
            silence_duration=data.get("recording", {}).get("silence_duration", 0.5),
            server_url=data.get("server", {}).get("url", "http://neuralforge:8000/api/v1"),
            llm_url=data.get("server", {}).get("llm_url", "http://neuralforge:8000/v1"),
            tts_model=data.get("server", {}).get("tts_model", "kokoro-v1"),
            tts_voice=data.get("server", {}).get("tts_voice", "af_bella"),
            stt_model=data.get("server", {}).get("stt_model", "Whisper-Tiny"),
            llm_model=data.get("server", {}).get("llm_model", "Qwen3.5-122B-A10B-GGUF"),
            max_retries=data.get("performance", {}).get("max_retries", 3),
            retry_delay=data.get("performance", {}).get("retry_delay", 1.0),
        )

    def save(self, config_path: str | None = None) -> None:
        """Save configuration to TOML file."""
        if config_path is None:
            config_path = os.path.expanduser("~/.gabbie/config.toml")

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "audio": {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "chunk_size": self.chunk_size,
            },
            "wake_word": {
                "model": self.wake_word,
                "threshold": self.detection_threshold,
                "vad_threshold": self.vad_threshold,
            },
            "recording": {
                "post_wake_buffer": self.post_wake_buffer,
                "silence_threshold": self.silence_threshold,
                "silence_duration": self.silence_duration,
            },
            "server": {
                "url": self.server_url,
                "llm_url": self.llm_url,
                "tts_model": self.tts_model,
                "tts_voice": self.tts_voice,
                "stt_model": self.stt_model,
                "llm_model": self.llm_model,
            },
            "performance": {
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
            },
        }

        if self.input_device_index is not None:
            data["audio"]["input_device_index"] = self.input_device_index
        if self.output_device_index is not None:
            data["audio"]["output_device_index"] = self.output_device_index

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "input_device_index": self.input_device_index,
            "output_device_index": self.output_device_index,
            "wake_word": self.wake_word,
            "detection_threshold": self.detection_threshold,
            "vad_threshold": self.vad_threshold,
            "post_wake_buffer": self.post_wake_buffer,
            "silence_threshold": self.silence_threshold,
            "silence_duration": self.silence_duration,
            "server_url": self.server_url,
            "llm_url": self.llm_url,
            "tts_model": self.tts_model,
            "tts_voice": self.tts_voice,
            "stt_model": self.stt_model,
            "llm_model": self.llm_model,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            chunk_size=data.get("chunk_size", 1280),
            input_device_index=data.get("input_device_index"),
            output_device_index=data.get("output_device_index"),
            wake_word=data.get("wake_word", "alexa"),
            detection_threshold=data.get("detection_threshold", 0.5),
            vad_threshold=data.get("vad_threshold", 0.5),
            post_wake_buffer=data.get("post_wake_buffer", 3),
            silence_threshold=data.get("silence_threshold", 0.01),
            silence_duration=data.get("silence_duration", 0.5),
            server_url=data.get("server_url", "http://neuralforge:8000/api/v1"),
            llm_url=data.get("llm_url", "http://neuralforge:8000/v1"),
            tts_model=data.get("tts_model", "kokoro-v1"),
            tts_voice=data.get("tts_voice", "af_bella"),
            stt_model=data.get("stt_model", "Whisper-Tiny"),
            llm_model=data.get("llm_model", "Qwen3.5-122B-A10B-GGUF"),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
        )
