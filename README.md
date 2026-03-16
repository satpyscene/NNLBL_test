# NNLBL: Neural Network Line-By-Line Absorption Calculator

A high-speed tool for computing atmospheric gas absorption cross-sections. NNLBL replaces traditional Voigt profile convolution with a Neural Network and uses GPU acceleration, achieving significant speedups while maintaining < 1% relative error compared to HAPI.

**Key features:**

- Hybrid HP/LP model switching based on Lorentz half-width (γ_L threshold)
- Full H₂O support including self-broadening and MT-CKD continuum absorption
- Single-layer (**SINGLE**) and vertical profile (**PROFILE**) modes
- Automatic CUDA detection

---

## Requirements

See `environment.yml`. Core dependencies: `torch`, `numpy`, `h5py`, `netCDF4`.

---

## Usage

All user-facing configuration lives in `example_config_NNLBL.py`. Edit it, then run:

```bash
python example_config_NNLBL.py
```

### Step 1 — Select target molecule and isotopes

```python
# HITRAN global isotope IDs, must all belong to the same molecule:
# H2O: [1,2,3,4,5,6,129]  CO2: [7-14]  O3: [16-20]  O2: [36-38]
TARGET_ISO_LIST = [1, 2]

ENABLE_CONTINUUM = False   # Set True only for H2O to include MT-CKD continuum
```

### Step 2 — Set the spectral grid

```python
SPECTRAL_CONFIG = {
    "min": 4800.0,   # cm⁻¹
    "max": 5200.0,
    "step": 0.01,
}
```

### Step 3 — Set the HP/LP model threshold

```python
# Lines with γ_L >= threshold → HP model (pressure-broadened)
# Lines with γ_L <  threshold → LP model (Doppler-dominated)
# Set to None to use the legacy pressure threshold (2200 Pa)
GAMMA_L_THRESHOLD = 0.001   # cm⁻¹
```

### Step 4 — Choose run mode and atmospheric state

**SINGLE** — one uniform atmospheric layer:

```python
RUN_MODE = "SINGLE"

SINGLE_PARAMS = {
    "p_hpa":    1013.25,   # Pressure     [hPa]
    "t_k":      296.0,     # Temperature  [K]
    "vmr_ppmv": 40000.0,   # VMR          [ppmv]
}
```

**PROFILE** — vertical atmospheric profile (one value per layer, consistent length across files):

```python
RUN_MODE = "PROFILE"

PROFILE_PARAMS = {
    "dir":      "atmospheric_profile_for_testing",
    "p_file":   "pres_100.txt",   "p_unit":   "hPa",    # or "Pa"
    "t_file":   "temp.txt",       "t_unit":   "K",      # or "C"
    "vmr_file": "h2o.txt",        "vmr_unit": "ppmv",   # or "vmr"
    "name_tag": "US_STD_100",
}
```

> **Profile layer ordering:** layer `000` = top of atmosphere (low pressure, sharp lines); layer `100` = near surface (high pressure, broad lines).

---

## Output

Results are written as `.h5` files to `sigma_output_filefold/`.

| Dataset | Contents |
| --- | --- |
| `wavenumber_grid` | Wavenumber axis (cm⁻¹) |
| `model_output/layer_NNN` | NNLBL cross-section (cm² molec.⁻¹) |
| `hapi_benchmark/layer_NNN` | HAPI reference cross-section (cm² molec.⁻¹) |

The `hapi_benchmark` group is only present when HAPI is run alongside NNLBL (see below).

---

## Reproducing Paper Results

Two scripts reproduce the NNLBL vs. HAPI comparison figures from the paper.

**Step 1 — Compute spectra** (run from project root):

```bash
python reproduce_paper_results.py            # NNLBL + HAPI benchmark
python reproduce_paper_results.py --skip-hapi  # NNLBL only (faster)
```

This computes cross-sections for O₃, H₂O, CO₂, and O₂ across multiple bands and atmospheric conditions, writing one `.h5` file per case to `sigma_output_filefold/`.

> **First run:** HAPI will download spectral line data from HITRAN and cache it to `data/`. Internet access required; may take a few minutes per molecule.

**Step 2 — Generate plots:**

```bash
python sigma_output_filefold/plot_results.py
```

Saves a two-panel PNG alongside each `.h5` file: top panel shows HAPI vs. NNLBL on a log scale; bottom panel shows absolute and relative error (%).

**Test cases covered:**

| Molecule | Band (cm⁻¹) | Step | Pressure | Temperature |
| --- | --- | --- | --- | --- |
| O₃ | 600–1200 | 0.01 | 1013.95 hPa | 279.1 K |
| O₃ | 600–1200 | 0.01 | 20 hPa | 220 K |
| O₃ | 1904–2566 | 0.01 / 0.001 | 7.56 hPa | 231.3 K |
| H₂O | 4100–4400 | 0.01 | 1013.95 hPa | 279.1 K |
| H₂O | 4100–4400 | 0.01 | 20 hPa | 220 K |
| CO₂ | 2150–2400 | 0.01 | 1013.95 hPa | 279.1 K |
| CO₂ | 2150–2400 | 0.01 | 20 hPa | 220 K |
| O₂ | 12975–13150 | 0.01 | 1013.95 hPa | 279.1 K |
| O₂ | 12975–13150 | 0.01 | 20 hPa | 220 K |
