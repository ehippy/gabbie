# gabbie

## Setup

```sh
uv run main.py
```

Available voices: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.

### Note on the vendored KittenTTS wheel

The upstream release wheel (`kittentts-0.8.0-py3-none-any.whl`) has a broken filename/metadata mismatch — its `.dist-info` internally claims version `0.1.0` while the filename says `0.8.0`, which `uv` refuses to resolve. [vendor/kittentts-0.8.0-py3-none-any.whl](vendor/kittentts-0.8.0-py3-none-any.whl) is the same wheel with the `.dist-info` directory renamed and `METADATA`/`RECORD` corrected to `0.8.0` so it's internally consistent. No functional code was changed.
