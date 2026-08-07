import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from kittentts import KittenTTS
import soundfile as sf


def main():
    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
    audio = model.generate(
        # ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
        "This high quality TTS model works without a GPU", voice="Luna"
    )
    sf.write("output.wav", audio, 24000)
    print("Wrote output.wav")


if __name__ == "__main__":
    main()
