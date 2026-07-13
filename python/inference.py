"""Live echolocation inference: emit chirp, process echo, classify material, update map."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from config import MATERIALS, MODEL_DIR, SAMPLE_RATE_HZ
from chirp import generate_lfm, synthetic_multichannel_echo
from echo import process_multichannel
from mapping import EchoMap
from model import load_model
from policy import select_action


def run_mapping_loop(
    model_path: str,
    max_steps: int = 50,
    synthetic: bool = True,
    device: str = "cpu",
) -> EchoMap:
    """
    Main exploration loop: policy -> chirp -> echo -> map update.

    Uses synthetic echoes when synthetic=True (no audio hardware).
    """
    model = load_model(model_path, device=device)
    echomap = EchoMap()
    last_glass_hint = False
    last_glass_conf = 0.0

    print("Starting EchoMap exploration loop...")
    for step in range(max_steps):
        action = select_action(echomap, last_glass_hint, last_glass_conf)
        if action is None:
            print(f"Map converged after {step} steps.")
            break

        print(f"step {step+1}: {action.chirp_mode}: {action.reason}")

        # Generate and "emit" chirp
        chirp_wav = generate_lfm(action.chirp_mode)

        if synthetic:
            # Simulate echo from a fake room
            range_m = 1.5 + (step % 3) * 0.5
            mat_idx = step % len(MATERIALS)
            recorded = synthetic_multichannel_echo(
                mode=action.chirp_mode,
                range_m=range_m,
                material_idx=mat_idx,
                bearing_deg=action.target_heading - echomap.robot_heading,
                seed=step * 42,
            )
        else:
            try:
                import sounddevice as sd

                recorded = sd.playrec(
                    chirp_wav,
                    samplerate=int(SAMPLE_RATE_HZ),
                    channels=1,
                    blocking=True,
                )
            except Exception as e:
                print(f"Audio I/O failed ({e}), falling back to synthetic.")
                recorded = synthetic_multichannel_echo(seed=step * 42)

        # Process echo
        result = process_multichannel(recorded, chirp_wav)

        # Update occupancy
        echomap.update_occupancy(
            result["range_m"],
            result["bearing_deg"],
            occupied=True,
        )

        # Classify material
        features = result["features"]
        feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat_t)
            probs = torch.softmax(logits, dim=1)[0]
            mat_idx = int(probs.argmax().item())
            conf = float(probs[mat_idx].item())

        echomap.update_material(
            result["range_m"],
            result["bearing_deg"],
            mat_idx,
            conf,
        )

        print(
            f"  range={result['range_m']:.2f}m  "
            f"bearing={result['bearing_deg']:.1f}deg  "
            f"material={MATERIALS[mat_idx]} ({conf:.2f})"
        )

        last_glass_hint = result["is_glass"]
        last_glass_conf = result["glass_confidence"]

        # Move robot toward target
        echomap.robot_x = action.target_x
        echomap.robot_y = action.target_y
        echomap.robot_heading = action.target_heading

    return echomap


def main() -> None:
    parser = argparse.ArgumentParser(description="Live EchoMap mapping loop")
    parser.add_argument("--model", default=str(MODEL_DIR / "echomap.pt"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--out", default="data/map_output.png")
    args = parser.parse_args()

    echomap = run_mapping_loop(
        args.model,
        max_steps=args.steps,
        synthetic=args.synthetic,
    )
    echomap.save_png(args.out)
    print(f"saved map -> {args.out}")


if __name__ == "__main__":
    main()
