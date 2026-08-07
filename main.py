from kittentts import KittenTTS
import soundfile as sf


def main():
    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
    audio = model.generate(
        "This high quality TTS model works without a GPU", voice="Jasper"
    )
    sf.write("output.wav", audio, 24000)
    print("Wrote output.wav")


if __name__ == "__main__":
    main()
