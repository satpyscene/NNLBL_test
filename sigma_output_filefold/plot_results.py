"""
Paper Results Reproduction — Visualization Script
Reads the 10 .h5 files under sigma_output_filefold/ and generates one PNG per file:
  Top panel:    HAPI vs NNLBL absorption cross-section comparison
  Bottom panel: Absolute error & relative error
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# Configuration
# ==============================================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files listed in display order with corresponding plot titles
FILES = [
    (
        "O3_Major_600.0_1200.0_0.01_101395_279.h5",
        "O$_3$  600–1200 cm$^{-1}$  |  1013.95 hPa, 279.1 K",
    ),
    (
        "O3_Major_600.0_1200.0_0.01_2000_220.h5",
        "O$_3$  600–1200 cm$^{-1}$  |  20 hPa, 220 K",
    ),
    (
        "H2O_Major_4100.0_4400.0_0.01_101395_279.h5",
        "H$_2$O  4100–4400 cm$^{-1}$  |  1013.95 hPa, 279.1 K",
    ),
    (
        "H2O_Major_4100.0_4400.0_0.01_2000_220.h5",
        "H$_2$O  4100–4400 cm$^{-1}$  |  20 hPa, 220 K",
    ),
    (
        "CO2_Major_2150.0_2400.0_0.01_101395_279.h5",
        "CO$_2$  2150–2400 cm$^{-1}$  |  1013.95 hPa, 279.1 K",
    ),
    (
        "CO2_Major_2150.0_2400.0_0.01_2000_220.h5",
        "CO$_2$  2150–2400 cm$^{-1}$  |  20 hPa, 220 K",
    ),
    (
        "O2_Major_12975.0_13150.0_0.01_101395_279.h5",
        "O$_2$  12975–13150 cm$^{-1}$  |  1013.95 hPa, 279.1 K",
    ),
    (
        "O2_Major_12975.0_13150.0_0.01_2000_220.h5",
        "O$_2$  12975–13150 cm$^{-1}$  |  20 hPa, 220 K",
    ),
    (
        "O3_Major_1904.0_2566.0_0.01_756_231.h5",
        "O$_3$  1904–2566 cm$^{-1}$  |  756.11 Pa, 231.295 K  (Δν = 0.01 cm$^{-1}$)",
    ),
    (
        "O3_Major_1904.0_2566.0_0.001_756_231.h5",
        "O$_3$  1904–2566 cm$^{-1}$  |  756.11 Pa, 231.295 K  (Δν = 0.001 cm$^{-1}$)",
    ),
]


# ==============================================================================
# Plotting
# ==============================================================================
def plot_one(fname, title, save_dir):
    fpath = os.path.join(save_dir, fname)
    if not os.path.exists(fpath):
        print(f"  [skip] file not found: {fname}")
        return

    with h5py.File(fpath, "r") as f:
        nn_sigma   = f["model_output/layer_000"][:]
        hapi_sigma = f["hapi_benchmark/layer_000"][:]
        wngrid     = f["wavenumber_grid"][:]

    abs_err = nn_sigma - hapi_sigma
    rel_err = (nn_sigma - hapi_sigma) / hapi_sigma * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Top panel: absorption cross-section
    ax1.plot(wngrid, hapi_sigma, lw=0.8, color="C0", label="HAPI (reference)", zorder=3)
    ax1.plot(wngrid, nn_sigma,   lw=0.8, color="C1", linestyle="--", label="NNLBL", zorder=4)
    ax1.set_yscale("log")
    ax1.set_ylabel("Absorption cross section\n(cm$^2$ molc.$^{-1}$)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # Bottom panel: errors
    ax2_r = ax2.twinx()
    ax2.plot(wngrid, abs_err, lw=0.7, color="C2", label="Absolute error", zorder=3)
    ax2_r.plot(wngrid, rel_err, lw=0.7, color="C3", linestyle="--",
               label="Relative error (%)", zorder=4)

    ax2.axhline(0,   color="gray", lw=0.6, linestyle=":")
    ax2_r.axhline(0, color="gray", lw=0.6, linestyle=":")

    ax2.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=10)
    ax2.set_ylabel("Absolute error\n(cm$^2$ molc.$^{-1}$)", color="C2", fontsize=9)
    ax2_r.set_ylabel("Relative error (%)", color="C3", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2_r.tick_params(axis="y", labelcolor="C3")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left",  fontsize=8)
    ax2_r.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    png_name  = fname.replace(".h5", ".png")
    save_path = os.path.join(save_dir, png_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  saved: {png_name}")
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}\n")
    for fname, title in FILES:
        print(f">>> {fname}")
        plot_one(fname, title, OUTPUT_DIR)
    print("\nAll plots generated.")
