#!/usr/bin/env python3
"""
Speech Enhancement — Facebook Demucs (State of the Art)
=========================================================
Uses Facebook Research's pretrained Demucs neural network.
Trained on DNS Challenge data — identifies and removes environmental
noise (wind, air, objects, breathing, hum) while preserving speech as-is.
No leveling, no compression, no gating — pure neural speech extraction.
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
import os
import time

INPUT_FILE = os.path.join(os.path.expanduser("~"), "Desktop",
                          "rapid reviw english part 6 .mp3")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "rapid_review_english_part6_cleaned.wav")


def main():
    print("\n" + "=" * 60)
    print("  SPEECH ENHANCEMENT — Facebook Demucs (SOTA)")
    print("  Neural network separates speech from environmental noise")
    print("=" * 60)

    t0 = time.time()

    # 1. Load audio
    print("\n  Loading audio...")
    from denoiser import pretrained, enhance
    import librosa
    y, sr = librosa.load(INPUT_FILE, sr=None, mono=True)
    wav = torch.from_numpy(y).float().unsqueeze(0)  # [1, samples]
    print(f"  Loaded: {wav.shape[1]/sr:.2f}s, {sr}Hz, channels={wav.shape[0]}")

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        print("  Converted to mono")

    # 2. Load pretrained Demucs model
    print("\n  Loading pretrained Demucs model...")
    model = pretrained.dns64()
    model.eval()

    # Resample to model's expected sample rate if needed
    model_sr = model.sample_rate
    if sr != model_sr:
        print(f"  Resampling {sr}Hz → {model_sr}Hz for model")
        wav = torchaudio.functional.resample(wav, sr, model_sr)

    # 3. Run denoising
    print("\n" + "=" * 60)
    print("  Running Demucs neural denoising...")
    print("  (identifies wind, air, object noise vs human speech)")
    print("=" * 60)

    with torch.no_grad():
        # Add batch dimension
        wav_input = wav.unsqueeze(0)  # [1, 1, samples]

        # Denoise
        enhanced = model(wav_input)

        # Remove batch dimension
        enhanced = enhanced.squeeze(0)  # [1, samples]

    # 4. Resample back to original SR if needed
    if sr != model_sr:
        print(f"  Resampling back {model_sr}Hz → {sr}Hz")
        enhanced = torchaudio.functional.resample(enhanced, model_sr, sr)

    # 5. Save
    enhanced_np = enhanced.squeeze().numpy()

    print(f"\n  Input  peak: {wav.abs().max():.4f}, RMS: {20*np.log10(wav.pow(2).mean().sqrt().item()+1e-10):.1f} dBFS")
    print(f"  Output peak: {np.max(np.abs(enhanced_np)):.4f}, RMS: {20*np.log10(np.sqrt(np.mean(enhanced_np**2))+1e-10):.1f} dBFS")

    sf.write(OUTPUT_FILE, enhanced_np, sr, subtype='PCM_24')
    sz = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n  Output: {OUTPUT_FILE}")
    print(f"  Size: {sz:.1f} MB")
    print(f"\n  DONE in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
