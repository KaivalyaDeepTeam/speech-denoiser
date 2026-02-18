# Speech Denoiser

Neural network-based speech enhancement that removes environmental noise (wind, air, breathing, hum, object sounds) while preserving speech exactly as-is.

Uses Facebook Research's pretrained **Demucs** model, trained on the [DNS Challenge](https://github.com/microsoft/DNS-Challenge) dataset.

## How It Works

The Demucs neural network has learned to distinguish between:
- **Human speech** — preserved without any distortion
- **Environmental noise** (wind, turbulence, breathing, hum, object sounds) — removed

No manual tuning, thresholds, or signal processing tricks. The model handles everything intelligently.

## Installation

```bash
pip install denoiser librosa soundfile torch torchaudio numpy
```

## Usage

1. Update `INPUT_FILE` and `OUTPUT_FILE` paths in `audio_preprocess.py`
2. Run:

```bash
python3 audio_preprocess.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `denoiser` | Facebook's Demucs speech enhancement model |
| `torch` | PyTorch (neural network backend) |
| `torchaudio` | Audio resampling |
| `librosa` | Audio loading |
| `soundfile` | WAV output |
| `numpy` | Numerical operations |

## Output

- Format: WAV, 24-bit PCM
- Sample rate: Same as input
- Channels: Mono

## References

- [Facebook Denoiser (Demucs)](https://github.com/facebookresearch/denoiser)
- [DNS Challenge](https://github.com/microsoft/DNS-Challenge)
- Defossez, A., Synnaeve, G., & Adi, Y. (2020). *Real Time Speech Enhancement in the Waveform Domain*. InterSpeech 2020.
