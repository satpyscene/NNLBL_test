# NNLBL — Paper Results Reproduction

This project computes atmospheric molecular infrared absorption spectra using the neural-network-based Voigt line shape model (NNLBL) and compares results quantitatively against the HAPI/HITRAN benchmark.

---

## Requirements

Python 3.8+, install dependencies:

```bash
pip install torch numpy h5py netCDF4 joblib tqdm matplotlib numba
```

Falls back to CPU automatically if no CUDA GPU is available.

---

## Directory Structure

```
NNLBL/
├── NNLBL_src/              # Core computation modules (inference engine, HAPI interface, MT-CKD model)
├── NNmodel&stats/          # Pre-trained model weights (.pth) and normalization statistics (.npy)
├── paper_results/
│   ├── main_test.py        # Entry script
│   ├── jqsrt_plot_utils.py # Plotting utilities
│   ├── data/               # HAPI spectral line cache (auto-downloaded on first run)
│   └── output/             # Generated comparison figures (auto-created)
└── README.md
```

---

## Usage

Run from the **project root directory**:

```bash
python paper_results/main_test.py
```

The script computes NNLBL vs. HAPI absorption spectra for four molecules (O₃, H₂O, CO₂, O₂) under two atmospheric conditions — high pressure (101.4 kPa, 279.1 K) and low pressure (2 kPa, 222 K) — and saves the comparison figures to `paper_results/output/`:

```
JQSRT_high_pressure_P101395_T279.png
JQSRT_low_pressure_P2000_T222.png
```

---

## Configuration

All parameters are defined at the top of `main_test.py`. Key settings:

|Parameter|Description|
|---|---|
|`HP_MODEL_PATH` / `LP_MODEL_PATH`|High-pressure / low-pressure model weight paths|
|`HP_STATS_PATH` / `LP_STATS_PATH`|Corresponding normalization statistics paths|
|`PRESSURE_THRESHOLD_PA`|Threshold for HP/LP model selection (default: 2200 Pa)|
|`TEST_CASES`|Test conditions (pressure, temperature, molecule, wavenumber range)|

---


