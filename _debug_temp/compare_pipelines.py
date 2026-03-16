"""
compare_pipelines.py — 对比两条计算流水线的误差来源

流水线 A（NNLBL_main，GPU float32）:
    来源: sigma_output_filefold/O2_Major_12975.0_13150.0_0.01_2000_220.h5

流水线 B（single_level_K_plot，CPU float64）:
    来源: paper_results/output_h5/O2_12975.0_13150.0_0.01_2000_220.h5

对比项目:
    1. HAPI_A  vs  HAPI_B   → 确认 HAPI 计算是否一致
    2. NNLBL_A vs  NNLBL_B  → 定量两条 NN 叠加流水线的差异
    3. NNLBL_A vs  HAPI_A   → 流水线 A 的 NN-HAPI 误差
    4. NNLBL_B vs  HAPI_B   → 流水线 B 的 NN-HAPI 误差

用法（从项目根目录运行）:
    python compare_pipelines.py
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 文件路径 ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

H5_A = os.path.join(ROOT, "sigma_output_filefold",
                    "O2_Major_12975.0_13150.0_0.01_2000_220.h5")
H5_B = os.path.join(ROOT, "paper_results", "output_h5",
                    "O2_12975.0_13150.0_0.01_2000_220.h5")

OUTPUT_PNG = os.path.join(ROOT, "pipeline_comparison_O2_lp.png")


# ── 读取数据 ──────────────────────────────────────────────────────────────
def load_pipeline_A(path):
    """从 NNLBL_main 输出的 h5 读取单层数据"""
    with h5py.File(path, "r") as f:
        wn     = f["wavenumber_grid"][:]
        nnlbl  = f["model_output/layer_000"][:]
        hapi   = f["hapi_benchmark/layer_000"][:]
    return wn, nnlbl.astype(np.float64), hapi.astype(np.float64)


def load_pipeline_B(path):
    """从 generate_figures 输出的 h5 读取数据"""
    with h5py.File(path, "r") as f:
        wn     = f["wavenumber_grid"][:]
        nnlbl  = f["nnlbl"][:]
        hapi   = f["hapi"][:]
    return wn, nnlbl.astype(np.float64), hapi.astype(np.float64)


# ── 相对误差（避免除零）────────────────────────────────────────────────────
def relative_error(pred, ref, eps_frac=1e-30):
    """(pred - ref) / max(|ref|, eps)，返回百分比"""
    eps = max(np.max(np.abs(ref)) * eps_frac, 1e-40)
    return (pred - ref) / (np.abs(ref) + eps) * 100.0


# ── 绘图 ──────────────────────────────────────────────────────────────────
def make_figure(wn, nnlbl_A, hapi_A, nnlbl_B, hapi_B):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        "O2  12975–13150 cm⁻¹  |  P=2000 Pa  T=220 K\n"
        "Pipeline A = NNLBL_main (GPU float32) | Pipeline B = single_level_K_plot (CPU float64)",
        fontsize=11,
    )

    # ── (1) 绝对光谱对比 ────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(wn, hapi_A,  lw=0.6, color="k",      label="HAPI_A")
    ax.semilogy(wn, hapi_B,  lw=0.6, color="gray",   label="HAPI_B", ls="--")
    ax.semilogy(wn, nnlbl_A, lw=0.6, color="tab:blue",  label="NNLBL_A")
    ax.semilogy(wn, nnlbl_B, lw=0.6, color="tab:orange", label="NNLBL_B", ls="--")
    ax.set_ylabel("Absorption [cm²/molec]")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("(1) Absolute spectra")

    # ── (2) HAPI_A vs HAPI_B — 核查 HAPI 一致性 ────────────────────────
    ax = axes[1]
    rel_hapi = relative_error(hapi_B, hapi_A)
    ax.plot(wn, rel_hapi, lw=0.5, color="tab:green")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Relative error [%]")
    ax.set_title(f"(2) HAPI_B vs HAPI_A  (max={np.max(np.abs(rel_hapi)):.3e} %,  "
                 f"mean|err|={np.mean(np.abs(rel_hapi)):.3e} %)")

    # ── (3) NNLBL_A vs NNLBL_B — 量化两条 NN 流水线差异 ────────────────
    ax = axes[2]
    rel_nn = relative_error(nnlbl_B, nnlbl_A)
    ax.plot(wn, rel_nn, lw=0.5, color="tab:blue")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Relative error [%]")
    ax.set_title(f"(3) NNLBL_B vs NNLBL_A  (max={np.max(np.abs(rel_nn)):.3e} %,  "
                 f"mean|err|={np.mean(np.abs(rel_nn)):.3e} %)")

    # ── (4) 两条流水线各自的 NNLBL-HAPI 误差 ──────────────────────────
    ax = axes[3]
    rel_A = relative_error(nnlbl_A, hapi_A)
    rel_B = relative_error(nnlbl_B, hapi_B)
    ax.plot(wn, rel_A, lw=0.5, color="tab:blue",   label=f"A: mean|err|={np.mean(np.abs(rel_A)):.3e} %")
    ax.plot(wn, rel_B, lw=0.5, color="tab:orange", label=f"B: mean|err|={np.mean(np.abs(rel_B)):.3e} %",
            ls="--")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Relative error [%]")
    ax.set_xlabel("Wavenumber [cm⁻¹]")
    ax.legend(fontsize=8)
    ax.set_title("(4) NNLBL vs HAPI per pipeline")

    plt.tight_layout()
    return fig


# ── 统计摘要 ──────────────────────────────────────────────────────────────
def print_summary(wn, nnlbl_A, hapi_A, nnlbl_B, hapi_B):
    rel_hapi = relative_error(hapi_B, hapi_A)
    rel_nn   = relative_error(nnlbl_B, nnlbl_A)
    rel_A    = relative_error(nnlbl_A, hapi_A)
    rel_B    = relative_error(nnlbl_B, hapi_B)

    print("\n" + "=" * 60)
    print("误差溯源统计摘要")
    print("=" * 60)
    fmt = "  {:<35s}  max={:.3e}%  mean|err|={:.3e}%"
    print(fmt.format("HAPI_B vs HAPI_A (HAPI一致性)",
                     np.max(np.abs(rel_hapi)), np.mean(np.abs(rel_hapi))))
    print(fmt.format("NNLBL_B vs NNLBL_A (流水线差异)",
                     np.max(np.abs(rel_nn)), np.mean(np.abs(rel_nn))))
    print(fmt.format("NNLBL_A vs HAPI_A (流水线A精度)",
                     np.max(np.abs(rel_A)), np.mean(np.abs(rel_A))))
    print(fmt.format("NNLBL_B vs HAPI_B (流水线B精度)",
                     np.max(np.abs(rel_B)), np.mean(np.abs(rel_B))))
    print("=" * 60)

    # 额外：找出 NNLBL_A vs NNLBL_B 差异最大的波数位置
    top_idx = np.argsort(np.abs(rel_nn))[-5:][::-1]
    print("\n  NNLBL_A vs NNLBL_B 差异最大的 5 个波数点:")
    for i in top_idx:
        print(f"    wn={wn[i]:.4f} cm⁻¹  rel_err={rel_nn[i]:.4e}%  "
              f"NNLBL_A={nnlbl_A[i]:.4e}  NNLBL_B={nnlbl_B[i]:.4e}")


# ── 主程序 ────────────────────────────────────────────────────────────────
def main():
    for label, path in [("A (NNLBL_main)", H5_A), ("B (generate_figures)", H5_B)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到流水线 {label} 的 h5 文件: {path}\n"
                "请先运行对应的计算脚本。"
            )

    print(f"读取流水线 A: {H5_A}")
    wn_A, nnlbl_A, hapi_A = load_pipeline_A(H5_A)

    print(f"读取流水线 B: {H5_B}")
    wn_B, nnlbl_B, hapi_B = load_pipeline_B(H5_B)

    # 对齐波数网格（两者可能因 np.arange 浮点差异导致长度差1，取公共范围）
    if len(wn_A) != len(wn_B):
        n = min(len(wn_A), len(wn_B))
        print(f"  [警告] 波数网格长度不同 (A={len(wn_A)}, B={len(wn_B)})，截取前 {n} 个点")
        wn_A, nnlbl_A, hapi_A = wn_A[:n], nnlbl_A[:n], hapi_A[:n]
        wn_B, nnlbl_B, hapi_B = wn_B[:n], nnlbl_B[:n], hapi_B[:n]

    if not np.allclose(wn_A, wn_B, atol=1e-4):
        raise ValueError("两条流水线的波数网格值不匹配，无法直接对比！")

    print_summary(wn_A, nnlbl_A, hapi_A, nnlbl_B, hapi_B)

    fig = make_figure(wn_A, nnlbl_A, hapi_A, nnlbl_B, hapi_B)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n  对比图已保存: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
