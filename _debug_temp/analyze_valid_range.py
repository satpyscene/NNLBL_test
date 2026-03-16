"""
Analyze the valid temperature and pressure range for the model
Based on training broadening parameter ranges, infer applicable T-P conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from NNLBL_src.run_inference_and_save import get_hapi_physical_params_new

# Training broadening parameter ranges
TRAIN_RANGES = {
    "gamma_d_min": 3.9e-4,  # cm^-1
    "gamma_d_max": 2.2e-2,  # cm^-1
    "gamma_l_low_pressure_min": 2.6e-8,  # cm^-1 (low pressure dataset)
    "gamma_l_low_pressure_max": 2.5e-3,  # cm^-1
    "gamma_l_high_pressure_min": 9.6e-5,  # cm^-1 (high pressure dataset)
    "gamma_l_high_pressure_max": 5.3e-1,  # cm^-1
}

# Combined gamma_l total range
GAMMA_L_MIN = TRAIN_RANGES["gamma_l_low_pressure_min"]
GAMMA_L_MAX = TRAIN_RANGES["gamma_l_high_pressure_max"]
GAMMA_D_MIN = TRAIN_RANGES["gamma_d_min"]
GAMMA_D_MAX = TRAIN_RANGES["gamma_d_max"]


def analyze_tp_range(
    molecule_name="CO2",
    global_iso_ids=[7],  # CO2 main isotopologue
    wavenumber_center=5000.0,  # Wavenumber center for calculation
    wavenumber_range=200.0,  # Wavenumber range
    T_range=(200, 400, 5),  # Temperature range (min, max, num_points)
    P_range=(100, 110000, 5),  # Pressure range in Pa (min, max, num_points)
    vmr=0.0,  # Volume mixing ratio
):
    """
    Scan T-P parameter space to determine conditions where γD and γL fall within training range

    Returns:
        T_grid, P_grid: Temperature and pressure grids
        validity_map: Validity map (0=invalid, 1=only γD valid, 2=only γL valid, 3=both valid)
        gamma_d_map, gamma_l_map: Distribution maps of broadening parameters
    """
    # Create T-P grid
    T_array = np.linspace(T_range[0], T_range[1], T_range[2])
    P_array = np.linspace(P_range[0], P_range[1], P_range[2])
    T_grid, P_grid = np.meshgrid(T_array, P_array)

    # Initialize result arrays
    validity_map = np.zeros_like(T_grid, dtype=int)
    gamma_d_mean_map = np.zeros_like(T_grid)
    gamma_l_mean_map = np.zeros_like(T_grid)
    gamma_d_range_map = np.zeros_like(T_grid)
    gamma_l_range_map = np.zeros_like(T_grid)

    wavenumber_min = wavenumber_center - wavenumber_range / 2
    wavenumber_max = wavenumber_center + wavenumber_range / 2

    print(
        f"Analyzing {molecule_name} in wavenumber range [{wavenumber_min}, {wavenumber_max}] cm^-1"
    )
    print(f"Scanning temperature range: {T_range[0]}-{T_range[1]} K")
    print(f"Scanning pressure range: {P_range[0]/100:.1f}-{P_range[1]/100:.1f} hPa")
    print("=" * 60)

    # Scan T-P space
    total = T_grid.size
    for idx, (T, P) in enumerate(zip(T_grid.flat, P_grid.flat)):
        if (idx + 1) % 100 == 0:
            print(f"Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")

        try:
            # Get physical parameters at this T,P condition
            params = get_hapi_physical_params_new(
                molecule_name=molecule_name,
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
                temperature_k=T,
                pressure_pa=P,
                global_iso_ids=global_iso_ids,
                vmr=vmr,
            )

            # Check if data exists
            if len(params["gamma_d"]) == 0:
                continue

            # Calculate broadening parameter ranges at this condition
            gamma_d_min_current = params["gamma_d"].min()
            gamma_d_max_current = params["gamma_d"].max()
            gamma_l_min_current = params["gamma_l"].min()
            gamma_l_max_current = params["gamma_l"].max()

            gamma_d_mean = params["gamma_d"].mean()
            gamma_l_mean = params["gamma_l"].mean()

            # Store statistics
            i, j = np.unravel_index(idx, T_grid.shape)
            gamma_d_mean_map[i, j] = gamma_d_mean
            gamma_l_mean_map[i, j] = gamma_l_mean
            gamma_d_range_map[i, j] = gamma_d_max_current - gamma_d_min_current
            gamma_l_range_map[i, j] = gamma_l_max_current - gamma_l_min_current

            # Check if within training range
            # Check if current broadening range overlaps with training range
            gamma_d_valid = (
                gamma_d_min_current <= GAMMA_D_MAX
                and gamma_d_max_current >= GAMMA_D_MIN
            )
            gamma_l_valid = (
                gamma_l_min_current <= GAMMA_L_MAX
                and gamma_l_max_current >= GAMMA_L_MIN
            )

            # Encode validity
            if gamma_d_valid and gamma_l_valid:
                validity_map[i, j] = 3  # Both valid
            elif gamma_d_valid:
                validity_map[i, j] = 1  # Only γD valid
            elif gamma_l_valid:
                validity_map[i, j] = 2  # Only γL valid

        except Exception as e:
            print(f"Warning: T={T}K, P={P}Pa processing failed: {e}")
            continue

    print("=" * 60)
    print("Analysis complete!")

    return T_grid, P_grid, validity_map, gamma_d_mean_map, gamma_l_mean_map


def plot_validity_map(
    T_grid,
    P_grid,
    validity_map,
    gamma_d_map,
    gamma_l_map,
    molecule_name,
    filename="model_validity_range.png",
):
    """Plot validity analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Validity heatmap
    ax1 = axes[0, 0]
    colors = ["white", "lightcoral", "lightblue", "lightgreen"]
    labels = ["Invalid", "Only γD Valid", "Only γL Valid", "Fully Valid"]

    im1 = ax1.contourf(
        T_grid,
        P_grid / 100,
        validity_map,
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
        colors=colors,
        alpha=0.8,
    )
    ax1.contour(
        T_grid,
        P_grid / 100,
        validity_map,
        levels=[0.5, 1.5, 2.5, 3.5],
        colors="black",
        linewidths=0.5,
    )
    ax1.set_xlabel("Temperature (K)", fontsize=12)
    ax1.set_ylabel("Pressure (hPa)", fontsize=12)
    ax1.set_title(f"{molecule_name} Model Valid Range", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # 2. γD mean value distribution
    ax2 = axes[0, 1]
    gamma_d_map_masked = np.ma.masked_where(validity_map == 0, gamma_d_map)
    im2 = ax2.contourf(
        T_grid, P_grid / 100, gamma_d_map_masked, levels=20, cmap="viridis"
    )
    plt.colorbar(im2, ax=ax2, label="Mean γD (cm$^{-1}$)")
    ax2.set_xlabel("Temperature (K)", fontsize=12)
    ax2.set_ylabel("Pressure (hPa)", fontsize=12)
    ax2.set_title("Doppler Broadening γD Mean Distribution", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add training range annotation
    ax2.text(
        0.02,
        0.98,
        f"Training Range:\n{GAMMA_D_MIN:.2e} - {GAMMA_D_MAX:.2e}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 3. γL mean value distribution
    ax3 = axes[1, 0]
    gamma_l_map_masked = np.ma.masked_where(validity_map == 0, gamma_l_map)
    im3 = ax3.contourf(
        T_grid,
        P_grid / 100,
        gamma_l_map_masked,
        levels=20,
        cmap="plasma",
        norm=plt.matplotlib.colors.LogNorm(),
    )
    plt.colorbar(im3, ax=ax3, label="Mean γL (cm$^{-1}$)")
    ax3.set_xlabel("Temperature (K)", fontsize=12)
    ax3.set_ylabel("Pressure (hPa)", fontsize=12)
    ax3.set_title("Lorentz Broadening γL Mean Distribution (log)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Add training range annotation
    ax3.text(
        0.02,
        0.98,
        f"Training Range:\n{GAMMA_L_MIN:.2e} - {GAMMA_L_MAX:.2e}",
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 4. Statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate statistics
    total_points = validity_map.size
    valid_points = np.sum(validity_map == 3)
    partial_valid = np.sum((validity_map == 1) | (validity_map == 2))
    invalid_points = np.sum(validity_map == 0)

    valid_ratio = valid_points / total_points * 100

    # Find fully valid T-P range
    valid_mask = validity_map == 3
    if valid_mask.any():
        valid_T = T_grid[valid_mask]
        valid_P = P_grid[valid_mask]
        T_min_valid, T_max_valid = valid_T.min(), valid_T.max()
        P_min_valid, P_max_valid = valid_P.min(), valid_P.max()
    else:
        T_min_valid = T_max_valid = P_min_valid = P_max_valid = 0

    stats_text = f"""
    Model Valid Range Statistical Report
    {'='*40}

    Molecule: {molecule_name}

    Scanning Range:
    • Temperature: {T_grid.min():.1f} - {T_grid.max():.1f} K
    • Pressure: {P_grid.min()/100:.1f} - {P_grid.max()/100:.1f} hPa

    Training Broadening Parameter Range:
    • γD: {GAMMA_D_MIN:.2e} - {GAMMA_D_MAX:.2e} cm⁻¹
    • γL: {GAMMA_L_MIN:.2e} - {GAMMA_L_MAX:.2e} cm⁻¹

    Validity Statistics:
    • Fully valid points: {valid_points}/{total_points} ({valid_ratio:.1f}%)
    • Partially valid points: {partial_valid}/{total_points} ({partial_valid/total_points*100:.1f}%)
    • Invalid points: {invalid_points}/{total_points} ({invalid_points/total_points*100:.1f}%)

    Recommended Valid Range (Fully Valid):
    • Temperature: {T_min_valid:.1f} - {T_max_valid:.1f} K
    • Pressure: {P_min_valid/100:.1f} - {P_max_valid/100:.1f} hPa
    """

    ax4.text(
        0.1,
        0.9,
        stats_text,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved as: {filename}")
    plt.close()


if __name__ == "__main__":
    # Define all molecules and their isotopologue IDs
    MOLECULES_CONFIG = {
        "H2O": [1, 2, 3, 4, 5, 6, 129],
        "CO2": [7, 8, 9, 10, 11, 12, 13, 14],
        "O3": [16, 17, 18, 19, 20],
        "N2O": [21, 22, 23, 24, 25],
        "CO": [26, 27, 28, 29, 30, 31],
        "CH4": [32, 33, 34, 35],
        "O2": [36, 37, 38],
    }

    # Analysis parameters (can be adjusted)
    ANALYSIS_CONFIG = {
        "wavenumber_center": 700.0,  # Wavenumber center
        "wavenumber_range": 100.0,  # Wavenumber range
        "T_range": (200, 400, 5),  # Temperature: 200-400K, 21 points
        "P_range": (100, 110000, 5),  # Pressure: 1-1100 hPa, 31 points
        "vmr": 0.0,  # Volume mixing ratio
    }

    print("=" * 70)
    print("Starting comprehensive analysis for all molecules")
    print(f"Total molecules to analyze: {len(MOLECULES_CONFIG)}")
    print("=" * 70)
    print()

    # Loop through all molecules
    for idx, (molecule_name, global_iso_ids) in enumerate(MOLECULES_CONFIG.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(MOLECULES_CONFIG)}] Analyzing molecule: {molecule_name}")
        print(f"Isotopologue IDs: {global_iso_ids}")
        print(f"{'='*70}\n")

        try:
            # Execute analysis
            T_grid, P_grid, validity_map, gamma_d_map, gamma_l_map = analyze_tp_range(
                molecule_name=molecule_name,
                global_iso_ids=global_iso_ids,
                wavenumber_center=ANALYSIS_CONFIG["wavenumber_center"],
                wavenumber_range=ANALYSIS_CONFIG["wavenumber_range"],
                T_range=ANALYSIS_CONFIG["T_range"],
                P_range=ANALYSIS_CONFIG["P_range"],
                vmr=ANALYSIS_CONFIG["vmr"],
            )

            # Plot results with molecule-specific filename
            plot_validity_map(
                T_grid,
                P_grid,
                validity_map,
                gamma_d_map,
                gamma_l_map,
                molecule_name,
                filename=f"model_validity_range_{molecule_name}.png",
            )

            print(f"\n✓ {molecule_name} analysis completed successfully!")

        except Exception as e:
            print(f"\n✗ Error analyzing {molecule_name}: {e}")
            print(f"Skipping {molecule_name} and continuing with next molecule...\n")
            continue

    print("\n" + "=" * 70)
    print("All analyses complete!")
    print("=" * 70)
    print("\nGenerated files:")
    for molecule_name in MOLECULES_CONFIG.keys():
        print(f"  - model_validity_range_{molecule_name}.png")
    print("\nRecommendations:")
    print("1. Check the generated figures for each molecule")
    print("2. Green regions indicate fully valid temperature-pressure ranges")
    print("3. Adjust ANALYSIS_CONFIG parameters for finer resolution if needed")
