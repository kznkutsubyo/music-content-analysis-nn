from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

IN_ROOT  = Path(r"..\data\gtzan\dataset\genres").resolve()
OUT_ROOT = Path(r"..\data\gtzan_wav\genres").resolve()

TARGET_SR = 16000

def convert_one(src: Path, dst: Path):
    # читаем .au через libsndfile (soundfile)
    audio, sr = sf.read(str(src), always_2d=True)  # shape: [T, C]
    audio = audio.astype(np.float32)

    # в mono
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)

    # numpy -> torch [1, T]
    wav = torch.from_numpy(audio.T)  # [C, T] где C=1

    # ресемпл до 16k
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    dst.parent.mkdir(parents=True, exist_ok=True)

    # сохраняем wav (PCM_16)
    sf.write(str(dst), wav.squeeze(0).cpu().numpy(), TARGET_SR, subtype="PCM_16")

def main():
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"IN_ROOT не существует: {IN_ROOT}")

    files = list(IN_ROOT.rglob("*.au"))
    if not files:
        raise FileNotFoundError(f"Не нашёл .au в {IN_ROOT}")

    print("IN :", IN_ROOT)
    print("OUT:", OUT_ROOT)
    print("files:", len(files))

    for i, src in enumerate(files, 1):
        rel = src.relative_to(IN_ROOT)
        dst = (OUT_ROOT / rel).with_suffix(".wav")
        convert_one(src, dst)
        if i % 100 == 0:
            print(f"converted {i}/{len(files)}")

    print("DONE")

if __name__ == "__main__":
    main()
