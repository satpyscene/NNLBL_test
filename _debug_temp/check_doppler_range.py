"""
Check Doppler broadening range for different molecules and wavenumber bands
Compare with training range to determine valid regions
"""

import numpy as np
from NNLBL_src.run_inference_and_save import get_hapi_physical_params_new

# Training Doppler broadening range
GAMMA_D_TRAIN_MIN = 3.9e-4  # cm^-1
GAMMA_D_TRAIN_MAX = 2.2e-2  # cm^-1

# Molecules and their isotopologue IDs
MOLECULES_CONFIG = {
    "H2O": [1, 2, 3, 4, 5, 6, 129],
    "CO2": [7, 8, 9, 10, 11, 12, 13, 14],
    "O3": [16, 17, 18, 19, 20],
    "N2O": [21, 22, 23, 24, 25],
    "CO": [26, 27, 28, 29, 30, 31],
    "CH4": [32, 33, 34, 35],
    "O2": [36, 37, 38],
}

# Wavenumber bands to test (cm^-1)
WAVENUMBER_BANDS = [
    # (500, 800, "Far-IR"),
    # (800, 1200, "Mid-IR 1"),
    # (1200, 2000, "Mid-IR 2"),
    # (2000, 3000, "Near-IR 1"),
    # (3000, 5000, "Near-IR 2"),
    # (5000, 8000, "Near-IR 3"),
    (13000, 13500, "Short-wave IR"),
]

# Temperature and pressure range for testing
T_TEST = [315]  # K
P_TEST = [101325]  # Pa


def calculate_doppler_range(molecule_name, global_iso_ids, wn_min, wn_max):
    """
    Calculate Doppler broadening range for a molecule in a wavenumber band

    Returns:
        gamma_d_min, gamma_d_max, is_valid
    """
    all_gamma_d = []

    for T in T_TEST:
        for P in P_TEST:
            try:
                params = get_hapi_physical_params_new(
                    molecule_name=molecule_name,
                    wavenumber_min=wn_min,
                    wavenumber_max=wn_max,
                    temperature_k=T,
                    pressure_pa=P,
                    global_iso_ids=global_iso_ids,
                    vmr=0.0,
                )

                if len(params["gamma_d"]) > 0:
                    all_gamma_d.extend(params["gamma_d"])

            except Exception as e:
                continue

    if len(all_gamma_d) == 0:
        return None, None, False

    gamma_d_min = np.min(all_gamma_d)
    gamma_d_max = np.max(all_gamma_d)

    # Check if within training range
    is_valid = gamma_d_min >= GAMMA_D_TRAIN_MIN and gamma_d_max <= GAMMA_D_TRAIN_MAX

    return gamma_d_min, gamma_d_max, is_valid


def main():
    print("=" * 100)
    print("Doppler Broadening Range Analysis")
    print(f"Training Range: {GAMMA_D_TRAIN_MIN:.2e} - {GAMMA_D_TRAIN_MAX:.2e} cm^-1")
    print(f"Testing T range: {min(T_TEST)}-{max(T_TEST)} K")
    print(f"Testing P range: {min(P_TEST)/100:.1f}-{max(P_TEST)/100:.1f} hPa")
    print("=" * 100)
    print()

    results = []

    for mol_name, iso_ids in MOLECULES_CONFIG.items():
        print(f"\nAnalyzing {mol_name}...")

        for wn_min, wn_max, band_name in WAVENUMBER_BANDS:
            print(f"  {band_name} ({wn_min}-{wn_max} cm^-1)...", end=" ")

            gamma_d_min, gamma_d_max, is_valid = calculate_doppler_range(
                mol_name, iso_ids, wn_min, wn_max
            )

            if gamma_d_min is None:
                print("No data")
                status = "No Data"
            else:
                status = "✓ Valid" if is_valid else "✗ Invalid"
                print(f"{status} | Min: {gamma_d_min:.2e}, Max: {gamma_d_max:.2e}")

                # Check if max exceeds training range
                if gamma_d_max > GAMMA_D_TRAIN_MAX:
                    exceed_ratio = (
                        (gamma_d_max - GAMMA_D_TRAIN_MAX) / GAMMA_D_TRAIN_MAX * 100
                    )
                    print(f"    ⚠ Max exceeds training by {exceed_ratio:.1f}%")

                if gamma_d_min < GAMMA_D_TRAIN_MIN:
                    exceed_ratio = (
                        (GAMMA_D_TRAIN_MIN - gamma_d_min) / GAMMA_D_TRAIN_MIN * 100
                    )
                    print(f"    ⚠ Min below training by {exceed_ratio:.1f}%")

            results.append(
                {
                    "Molecule": mol_name,
                    "Band": band_name,
                    "Wavenumber": f"{wn_min}-{wn_max}",
                    "Min γD": gamma_d_min,
                    "Max γD": gamma_d_max,
                    "Status": status,
                }
            )

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Molecule':<8} {'Band':<15} {'Wavenumber':<15} {'Min γD':<12} {'Max γD':<12} {'Status':<10}"
    )
    print("-" * 100)

    for r in results:
        min_str = f"{r['Min γD']:.2e}" if r["Min γD"] is not None else "N/A"
        max_str = f"{r['Max γD']:.2e}" if r["Max γD"] is not None else "N/A"
        print(
            f"{r['Molecule']:<8} {r['Band']:<15} {r['Wavenumber']:<15} {min_str:<12} {max_str:<12} {r['Status']:<10}"
        )

    print("=" * 100)

    # Count valid combinations
    valid_count = sum(1 for r in results if r["Status"] == "✓ Valid")
    total_count = len([r for r in results if r["Status"] != "No Data"])

    print(f"\nValid combinations: {valid_count}/{total_count}")
    print(f"Coverage: {valid_count/total_count*100:.1f}%")

    # Save to CSV
    import csv

    with open("doppler_range_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Molecule", "Band", "Wavenumber", "Min γD", "Max γD", "Status"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to: doppler_range_analysis.csv")


if __name__ == "__main__":
    main()
