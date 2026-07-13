"""Shared constants for EchoMap pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = DATA_DIR / "models"

# Audio
SAMPLE_RATE_HZ = 48000
SPEED_OF_SOUND_M_S = 343.0
NUM_MICS = 4
MIC_SPACING_M = 0.03  # 3 cm linear array spacing

# Chirp modes
CHIRP_MODES = ["GEOMETRY", "MATERIAL", "GLASS_PROBE"]
CHIRP_PARAMS = {
    "GEOMETRY": {"f0": 2000.0, "f1": 8000.0, "duration_ms": 15.0},
    "MATERIAL": {"f0": 8000.0, "f1": 20000.0, "duration_ms": 8.0},
    "GLASS_PROBE": {"f0": 12000.0, "f1": 24000.0, "duration_ms": 8.0},
}

# Materials
MATERIALS = [
    "drywall",
    "wood",
    "glass",
    "metal",
    "carpet",
    "concrete",
]
NUM_CLASSES = len(MATERIALS)
MATERIAL_TO_IDX = {name: i for i, name in enumerate(MATERIALS)}

# Map
MAP_RESOLUTION_M = 0.05  # 5 cm cells
MAP_SIZE_M = 6.0  # 6 m x 6 m grid
MAP_CELLS = int(MAP_SIZE_M / MAP_RESOLUTION_M)

# Policy thresholds
MATERIAL_CONFIDENCE_THRESHOLD = 0.70
MATERIAL_STOP_THRESHOLD = 0.85
GLASS_KURTOSIS_THRESHOLD = 4.0
APPROACH_DISTANCE_M = 0.75

# Feature extraction
MFCC_COEFFS = 13
MFCC_NFFT = 512
MFCC_HOP = 256

# Robot odometry (from firmware CSV)
TICKS_PER_REV = 20
WHEEL_DIAMETER_M = 0.065
WHEEL_BASE_M = 0.14
