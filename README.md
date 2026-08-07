# gabbie

## Setup

The KittenTTS release wheel has a filename/metadata version mismatch (`kittentts-0.8.0-py3-none-any.whl` reports version `0.1.0`), so `uv` needs to be told to tolerate it:

```sh
export UV_SKIP_WHEEL_FILENAME_CHECK=1
uv run main.py
```

Available voices: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.
