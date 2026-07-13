"""Record labeled echo samples for material classifier training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from config import MATERIAL_TO_IDX, RAW_DIR, SAMPLE_RATE_HZ
from chirp import generate_lfm


def record_echo(
    material: str,
    mode: str = "MATERIAL",
    duration_sec: float = 0.5,
    fs: float = SAMPLE_RATE_HZ,
    use_synthetic: bool = False,
) -> np.ndarray:
    """
    Record echo response for a material panel.

    Falls back to synthetic echo when sounddevice is unavailable or
    use_synthetic=True.
    """
    if material not in MATERIAL_TO_IDX:
        raise ValueError(f"Unknown material. Choose from: {list(MATERIAL_TO_IDX)}")

    chirp_wav = generate_lfm(mode, fs=fs)
    record_len = int((duration_sec + 0.1) * fs)

    if use_synthetic:
        from chirp import synthetic_multichannel_echo

        mat_idx = MATERIAL_TO_IDX[material]
        return synthetic_multichannel_echo(
            mode=mode,
            range_m=1.0,
            material_idx=mat_idx,
            seed=hash(material) % 10000,
        )

    try:
        import sounddevice as sd

        print(f"Emitting {mode} chirp toward '{material}' panel...")
        recorded = sd.playrec(
            chirp_wav,
            samplerate=int(fs),
            channels=1,
            blocking=True,
        )
        # Pad to expected length
        if len(recorded) < record_len:
            recorded = np.pad(recorded, (0, record_len - len(recorded)))
        return recorded[:record_len].astype(np.float64)
    except Exception as e:
        print(f"Audio I/O unavailable ({e}), using synthetic echo.")
        from chirp import synthetic_multichannel_echo

        mat_idx = MATERIAL_TO_IDX[material]
        return synthetic_multichannel_echo(
            mode=mode,
            range_m=1.0,
            material_idx=mat_idx,
            seed=hash(material) % 10000,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record echo for one material label")
    parser.add_argument("--material", required=True, help="e.g. drywall, wood, glass")
    parser.add_argument("--mode", default="MATERIAL", help="Chirp mode")
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--synthetic", action="store_true", help="Use fake echoes")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = record_echo(
        args.material,
        mode=args.mode,
        duration_sec=args.duration,
        use_synthetic=args.synthetic,
    )
    ts = int(time.time())
    out_path = args.out_dir / f"{args.material}_{args.mode}_{ts}.npy"
    np.save(out_path, data)
    print(f"saved {data.shape} -> {out_path}")


if __name__ == "__main__":
    main()
