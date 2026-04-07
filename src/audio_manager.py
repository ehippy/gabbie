"""Audio device management for Gabbie."""

import logging
import sys
from dataclasses import dataclass
from typing import Any

import pyaudio

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Audio device information."""

    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False


class AudioManager:
    """Manages audio device enumeration and selection."""

    def __init__(self):
        self.paudio = pyaudio.PyAudio()
        self._input_devices: list[AudioDevice] = []
        self._output_devices: list[AudioDevice] = []
        self._default_input: int = 0
        self._default_output: int = 0
        self._enumerate_devices()

    def _enumerate_devices(self) -> None:
        """Enumerate all available audio devices."""
        input_devices = []
        output_devices = []

        try:
            default_input_info = self.paudio.get_default_input_device_info()
            self._default_input = int(default_input_info["index"])
        except Exception:
            self._default_input = 0

        try:
            default_output_info = self.paudio.get_default_output_device_info()
            self._default_output = int(default_output_info["index"])
        except Exception:
            self._default_output = 0

        for i in range(self.paudio.get_device_count()):
            try:
                info = self.paudio.get_device_info_by_index(i)
                name = str(info.get("name", f"Device {i}"))
                max_input = int(info.get("maxInputChannels", 0))
                max_output = int(info.get("maxOutputChannels", 0))
                default_rate = int(info.get("defaultSampleRate", 44100))

                if max_input > 0:
                    input_devices.append(
                        AudioDevice(
                            index=i,
                            name=name,
                            channels=max_input,
                            sample_rate=default_rate,
                            is_default=(i == self._default_input),
                        )
                    )

                if max_output > 0:
                    output_devices.append(
                        AudioDevice(
                            index=i,
                            name=name,
                            channels=max_output,
                            sample_rate=default_rate,
                            is_default=(i == self._default_output),
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not get info for device {i}: {e}")

        self._input_devices = input_devices
        self._output_devices = output_devices

    @property
    def input_devices(self) -> list[AudioDevice]:
        """Get list of input devices."""
        return self._input_devices

    @property
    def output_devices(self) -> list[AudioDevice]:
        """Get list of output devices."""
        return self._output_devices

    @property
    def default_input_index(self) -> int:
        """Get default input device index."""
        return self._default_input

    @property
    def default_output_index(self) -> int:
        """Get default output device index."""
        return self._default_output

    def get_input_device(self, index: int | None = None) -> AudioDevice | None:
        """Get input device by index, or default if None."""
        if index is None:
            index = self._default_input
        for device in self._input_devices:
            if device.index == index:
                return device
        return None

    def get_output_device(self, index: int | None = None) -> AudioDevice | None:
        """Get output device by index, or default if None."""
        if index is None:
            index = self._default_output
        for device in self._output_devices:
            if device.index == index:
                return device
        return None

    def list_devices(self) -> dict[str, Any]:
        """List all audio devices in a structured format."""
        return {
            "input": [
                {
                    "index": d.index,
                    "name": d.name,
                    "channels": d.channels,
                    "sample_rate": d.sample_rate,
                    "is_default": d.is_default,
                }
                for d in self._input_devices
            ],
            "output": [
                {
                    "index": d.index,
                    "name": d.name,
                    "channels": d.channels,
                    "sample_rate": d.sample_rate,
                    "is_default": d.is_default,
                }
                for d in self._output_devices
            ],
            "default_input": self._default_input,
            "default_output": self._default_output,
        }

    def close(self) -> None:
        """Clean up PyAudio resources."""
        try:
            self.paudio.terminate()
        except Exception:
            pass

    def __enter__(self) -> "AudioManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()
