"""
Paper Results Reproduction — Data Generation Script
Usage: python reproduce_paper_results.py [--skip-hapi]
"""

import sys
from pathlib import Path
from NNLBL_src.NNLBL_main import NNLBL_API, validate_user_config

# ==============================================================================
# Common configuration
# ==============================================================================
PATH_CONFIG = {
    "base_dir": Path(__file__).parent,
    "model_dir": "NNmodel&stats",
    "output_dir": "sigma_output_filefold",
    "mtckd_file": "data/absco-ref_wv-mt-ckd.nc",
}

GAMMA_L_THRESHOLD = None  # consistent with training boundaries
SKIP_HAPI = "--skip-hapi" in sys.argv

# ==============================================================================
# Task definitions
# ==============================================================================
# HITRAN global isotope IDs (primary isotopologues):
#   H2O → 1,  CO2 → 7,  O3 → 16,  O2 → 36

# Atmospheric conditions
COND_LOW_ALT = {"p_hpa": 1013.95, "t_k": 279.1, "vmr_ppmv": 0.0}   # near surface
COND_HIGH_ALT = {"p_hpa": 20.0,   "t_k": 220.0, "vmr_ppmv": 0.0}   # ~20 hPa
COND_O3_SPEC = {
    "p_hpa": 756.11 / 100.0,  # 756.11 Pa → hPa
    "t_k": 231.295,
    "vmr_ppmv": 0.0,
}  # O3 special condition

TASKS = [
    # ── O3  600–1200 cm⁻¹ ─────────────────────────────────────────────────
    {
        "label": "O3_600-1200_LOW-ALT",
        "iso_list": [16],
        "spectral": {"min": 600.0, "max": 1200.0, "step": 0.01},
        "single": COND_LOW_ALT,
        "continuum": False,
    },
    {
        "label": "O3_600-1200_HIGH-ALT",
        "iso_list": [16],
        "spectral": {"min": 600.0, "max": 1200.0, "step": 0.01},
        "single": COND_HIGH_ALT,
        "continuum": False,
    },
    # ── H2O 4100–4400 cm⁻¹ ────────────────────────────────────────────────
    {
        "label": "H2O_4100-4400_LOW-ALT",
        "iso_list": [1],
        "spectral": {"min": 4100.0, "max": 4400.0, "step": 0.01},
        "single": COND_LOW_ALT,
        "continuum": False,
    },
    {
        "label": "H2O_4100-4400_HIGH-ALT",
        "iso_list": [1],
        "spectral": {"min": 4100.0, "max": 4400.0, "step": 0.01},
        "single": COND_HIGH_ALT,
        "continuum": False,
    },
    # ── CO2 2150–2400 cm⁻¹ ────────────────────────────────────────────────
    {
        "label": "CO2_2150-2400_LOW-ALT",
        "iso_list": [7],
        "spectral": {"min": 2150.0, "max": 2400.0, "step": 0.01},
        "single": COND_LOW_ALT,
        "continuum": False,
    },
    {
        "label": "CO2_2150-2400_HIGH-ALT",
        "iso_list": [7],
        "spectral": {"min": 2150.0, "max": 2400.0, "step": 0.01},
        "single": COND_HIGH_ALT,
        "continuum": False,
    },
    # ── O2  12975–13150 cm⁻¹ ──────────────────────────────────────────────
    {
        "label": "O2_12975-13150_LOW-ALT",
        "iso_list": [36],
        "spectral": {"min": 12975.0, "max": 13150.0, "step": 0.01},
        "single": COND_LOW_ALT,
        "continuum": False,
    },
    {
        "label": "O2_12975-13150_HIGH-ALT",
        "iso_list": [36],
        "spectral": {"min": 12975.0, "max": 13150.0, "step": 0.01},
        "single": COND_HIGH_ALT,
        "continuum": False,
    },
    # ── O3  1904–2566 cm⁻¹, 756.11 Pa / 231.295 K, two resolutions ───────
    {
        "label": "O3_1904-2566_756Pa_step0.01",
        "iso_list": [16],
        "spectral": {"min": 1904.0, "max": 2566.0, "step": 0.01},
        "single": COND_O3_SPEC,
        "continuum": False,
    },
    {
        "label": "O3_1904-2566_756Pa_step0.001",
        "iso_list": [16],
        "spectral": {"min": 1904.0, "max": 2566.0, "step": 0.001},
        "single": COND_O3_SPEC,
        "continuum": False,
    },
]

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    total = len(TASKS)
    for idx, task in enumerate(TASKS, start=1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}]  {task['label']}")
        print(f"{'='*70}")

        validate_user_config(
            run_mode="SINGLE",
            single_params=task["single"],
            profile_params={},  # not used in SINGLE mode
            spectral_config=task["spectral"],
            target_iso_list=task["iso_list"],
        )

        NNLBL_API(
            target_iso_list=task["iso_list"],
            spectral_config=task["spectral"],
            input_mode="SINGLE",
            single_config=task["single"],
            profile_config={},
            path_config=PATH_CONFIG,
            enable_continuum=task["continuum"],
            skip_hapi=SKIP_HAPI,
            gamma_l_threshold=GAMMA_L_THRESHOLD,
        )

    print(f"\n{'='*70}")
    print("All tasks completed. Output files saved to:", PATH_CONFIG["output_dir"])
    print(f"{'='*70}")
