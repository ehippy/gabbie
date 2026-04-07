#!/usr/bin/env python3
"""Test openwakeword with your microphone."""

import pyaudio
import numpy as np
from openwakeword import Model
import time


def main():
    print("=" * 50)
    print("OpenWakeWord Test")
    print("=" * 50)

    print("\nLoading model...")
    model = Model()
    print(f"Models loaded: {list(model.models.keys())}")

    p = pyaudio.PyAudio()

    # List input devices
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            default = (
                " [DEFAULT]" if i == p.get_default_input_device_info()["index"] else ""
            )
            print(f"  {i}: {info['name']} ({info['defaultSampleRate']}Hz){default}")

    # Choose device - change this to your mic index
    device_index = 7  # Logitech BRIO
    try:
        device_info = p.get_device_info_by_index(device_index)
        print(f"\nUsing device: {device_info['name']}")
    except:
        print(f"\nDevice {device_index} not found, using default")
        device_index = p.get_default_input_device_info()["index"]

    # Open stream at 16kHz (what openwakeword expects)
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=int(device_index),
        frames_per_buffer=1280,
    )

    print("\n" + "-" * 50)
    print("Listening for 10 seconds...")
    print("Say 'Alexa', 'Hey Mycroft', 'Hey Jarvis' clearly!")
    print("-" * 50 + "\n")

    start = time.time()
    frame_count = 0
    max_scores = {"alexa": 0, "hey_mycroft": 0, "hey_jarvis": 0}

    try:
        while time.time() - start < 10:
            audio_data = stream.read(1280, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            predictions = model.predict(audio_array)

            frame_count += 1

            # Print every 20 frames (about 1 second)
            if frame_count % 20 == 0:
                alexa = predictions.get("alexa", 0)
                mycroft = predictions.get("hey_mycroft", 0)
                jarvis = predictions.get("hey_jarvis", 0)

                max_scores["alexa"] = max(max_scores["alexa"], alexa)
                max_scores["hey_mycroft"] = max(max_scores["hey_mycroft"], mycroft)
                max_scores["hey_jarvis"] = max(max_scores["hey_jarvis"], jarvis)

                print(
                    f"{frame_count // 20}s: alexa={alexa:.4f}  mycroft={mycroft:.4f}  jarvis={jarvis:.4f}"
                )

                if alexa > 0.1 or mycroft > 0.1 or jarvis > 0.1:
                    print("  ^^^ DETECTION!")

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        stream.close()
        p.terminate()

    print("\n" + "=" * 50)
    print("SUMMARY - Maximum scores:")
    print(f"  alexa:      {max_scores['alexa']:.4f}")
    print(f"  hey_mycroft: {max_scores['hey_mycroft']:.4f}")
    print(f"  hey_jarvis:  {max_scores['hey_jarvis']:.4f}")
    print("=" * 50)

    if max(max_scores.values()) < 0.01:
        print("\nWARNING: No significant detections!")
        print("Possible causes:")
        print("  - Microphone volume too low")
        print("  - Wake word models don't match your voice/accent")
        print("  - Background noise interfering")
        print("\nTry:")
        print("  - Speaking louder and closer to mic")
        print("  - Using headphones to reduce echo")
        print("  - Checking mic levels in system settings")


if __name__ == "__main__":
    main()
