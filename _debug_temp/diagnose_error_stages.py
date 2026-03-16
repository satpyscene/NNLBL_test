"""
diagnose_error_stages.py — 逐阶段消融分析
确认两条流水线的误差究竟来自哪几个阶段

策略：以流水线 B（CPU float64）为参考基线。
从 B 的中间结果出发，逐步替换成 A 的实现，
观察每一步替换引入的误差贡献。

Stage 0  (参考) : B 的完整流水线（float64 全程）
Stage 1  (消融1): B 谱线参数 → B 归一化 → B NN推理 → B 物理还原 → A 叠加(float32)
Stage 2  (消融2): B 谱线参数 → B 归一化 → B NN推理 → A 物理还原(float32) → A 叠加(float32)
Stage 3  (消融3): B 谱线参数 → A 归一化+NN+物理还原(float32 全程) → A 叠加(float32)
Stage 4  (消融4): A 谱线参数(float32) → A 归一化+NN+物理还原(float32 全程) → A 叠加(float32)
         [这应该与流水线 A 的实际结果吻合]

用法（从项目根目录运行）:
    python diagnose_error_stages.py
"""
import sys, os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
# single_level_K_plot.py 用裸导入 "from hapi import"，需要将含 hapi.py 的目录加入路径
sys.path.insert(0, os.path.join(ROOT_DIR, "paper_results"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ── 导入两条流水线的函数 ────────────────────────────────────────────────────
from NNLBL_src.single_level_K_plot import (
    get_hapi_physical_params   as get_params_B,   # float64
    create_non_uniform_grid,
    physical_restore_numpy,
    perform_superposition_cpu_safe,
    load_model                 as load_model_B,
)
from NNLBL_src.run_inference_and_save import (
    get_hapi_physical_params_new as get_params_A,  # float32
    perform_superposition_gpu,
)

# ── 实验条件 ─────────────────────────────────────────────────────────────────
MOLECULE    = "O2"
WN_MIN, WN_MAX, WN_STEP = 12975.0, 13150.0, 0.01
PRESSURE_PA = 2000.0
TEMP_K      = 220.0
GLOBAL_ISO_IDS = [36]      # O2 主同位素

ROOT = os.path.dirname(os.path.abspath(__file__))
LP_MODEL_PATH = os.path.join(ROOT, "NNmodel&stats/voigt_model_best_lp_Full-nonuniform-n0_1000_noshift.pth")
LP_STATS_PATH = os.path.join(ROOT, "NNmodel&stats/lp_voigt_stats_Full-nonuniform-n0_1000_noshift.npy")

OUTPUT_PNG = os.path.join(ROOT, "error_stage_ablation_O2_lp.png")

WING_SIZE = 25.0
TOTAL_POINTS = 5001
CONCENTRATION_FACTOR = 6.0

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
def relative_error_pct(pred, ref):
    eps = np.max(np.abs(ref)) * 1e-30 + 1e-40
    return (pred - ref) / (np.abs(ref) + eps) * 100.0

def stats_str(arr):
    return (f"max={np.max(np.abs(arr)):.3e}%  "
            f"mean|e|={np.mean(np.abs(arr)):.3e}%  "
            f"rms={np.sqrt(np.mean(arr**2)):.3e}%")

def physical_restore_float32(pred_norm_f64, y_mean, y_std, S_f64, nu0_f64, delta0_f64, base_wn):
    """用 float32 做物理还原（模拟 A 的 forward_with_full_pipeline）"""
    pred_norm = pred_norm_f64.astype(np.float32)
    y_m = np.float32(y_mean)
    y_s = np.float32(y_std)
    S   = S_f64.astype(np.float32)
    nu0 = nu0_f64.astype(np.float32)
    d0  = delta0_f64.astype(np.float32)
    base = base_wn.astype(np.float32)

    y_log = pred_norm * y_s + y_m                         # float32
    y_phys = (10.0 ** y_log.astype(np.float64)).astype(np.float32)  # pow 用 float32 精度
    profiles = (y_phys * S[:, None]).astype(np.float32)
    shifts = ((nu0 + d0) - np.float32(1000.0)).astype(np.float32)
    wn_grids = (base[None, :] + shifts[:, None]).astype(np.float32)
    return profiles, wn_grids

def superposition_A(profiles_f32, wn_grids_f32, global_wn, base_wn):
    """用 A 的 GPU float32 叠加"""
    device = torch.device("cpu")
    p_t = torch.from_numpy(profiles_f32.astype(np.float32)).to(device)
    w_t = torch.from_numpy(wn_grids_f32.astype(np.float32)).to(device)
    return perform_superposition_gpu(p_t, w_t, global_wn.astype(np.float64),
                                     base_wn.astype(np.float64))

def superposition_B(profiles_f64, wn_grids_f64, global_wn):
    """用 B 的 CPU float64 叠加"""
    return perform_superposition_cpu_safe(
        profiles_f64.astype(np.float64),
        wn_grids_f64.astype(np.float64),
        global_wn.astype(np.float64),
    )

# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cpu")
    stats = np.load(LP_STATS_PATH, allow_pickle=True).item()

    # 公共网格
    global_wn = np.arange(WN_MIN, WN_MAX, WN_STEP)   # B 的网格（17500 点）
    base_wn   = create_non_uniform_grid(1000.0, WING_SIZE, TOTAL_POINTS, CONCENTRATION_FACTOR)

    # ── 获取谱线参数（两个版本）────────────────────────────────────────────
    print("获取谱线参数 B (float64)...")
    params_B_list = get_params_B(MOLECULE, WN_MIN, WN_MAX, TEMP_K, PRESSURE_PA)
    # params_B_list 是 list of dicts
    gamma_d_B = np.array([p["gamma_d"] for p in params_B_list], dtype=np.float64)
    gamma_l_B = np.array([p["gamma_l"] for p in params_B_list], dtype=np.float64)
    S_B       = np.array([p["S"]       for p in params_B_list], dtype=np.float64)
    nu0_B     = np.array([p["nu0"]     for p in params_B_list], dtype=np.float64)
    delta0_B  = np.array([p["delta_0"] for p in params_B_list], dtype=np.float64)

    print("获取谱线参数 A (float32)...")
    params_A = get_params_A(MOLECULE, WN_MIN, WN_MAX, TEMP_K, PRESSURE_PA,
                             global_iso_ids=GLOBAL_ISO_IDS, vmr=0.0)
    gamma_d_A = params_A["gamma_d"].astype(np.float64)   # 提升到f64以便比较
    gamma_l_A = params_A["gamma_l"].astype(np.float64)
    S_A       = params_A["S"].astype(np.float64)
    nu0_A     = params_A["nu0"].astype(np.float64)
    delta0_A  = params_A["delta_0"].astype(np.float64)

    print(f"\n谱线数: B={len(gamma_d_B)}, A={len(gamma_d_A)}")
    print("谱线参数差异（B和A之间）:")
    for name, b, a in [("gamma_d", gamma_d_B, gamma_d_A),
                        ("gamma_l", gamma_l_B, gamma_l_A),
                        ("S",       S_B,       S_A),
                        ("nu0",     nu0_B,     nu0_A),
                        ("delta_0", delta0_B,  delta0_A)]:
        if len(b) == len(a):
            rd = np.abs((a - b) / (np.abs(b) + 1e-40)) * 100
            print(f"  {name:8s}: max_rel={np.max(rd):.3e}%  mean_rel={np.mean(rd):.3e}%")
        else:
            print(f"  {name:8s}: 长度不同 B={len(b)} A={len(a)}")

    # ── 模型推理（公用，用 B 的 float64 参数，float32 NN）────────────────────
    model_B = load_model_B(LP_MODEL_PATH, 2, [100,500,1000,500], TOTAL_POINTS, device)

    def run_nn_B_params(gd, gl, stats):
        """用 B 的 float64 线参数，通过 B 的模型推理流程（float32 NN + float64 归一化）"""
        inputs_raw = np.stack([gd, gl], axis=1)
        inputs_norm = (inputs_raw - stats["x_mean"]) / stats["x_std"]   # float64
        tensor_in = torch.tensor(inputs_norm, dtype=torch.float32)
        with torch.no_grad():
            pred_norm = model_B(tensor_in).cpu().numpy()   # float32 out
        return pred_norm  # shape (N, 5001)

    print("\n运行 NN 推理（B 参数）...")
    pred_norm_Bparams = run_nn_B_params(gamma_d_B, gamma_l_B, stats)

    # ── Stage 0: B 完整流水线（参考）────────────────────────────────────────
    print("\n[Stage 0] B 完整 float64 流水线...")
    profiles_B, wn_grids_B = physical_restore_numpy(
        pred_norm_Bparams, stats["y_mean"], stats["y_std"],
        S_B, nu0_B, delta0_B, base_wn)
    absorp_B = superposition_B(profiles_B, wn_grids_B, global_wn)

    # ── Stage 1: 仅替换叠加为 A 的 float32 ──────────────────────────────────
    print("[Stage 1] 替换叠加 → float32 (A)...")
    absorp_S1 = superposition_A(
        profiles_B.astype(np.float32),
        wn_grids_B.astype(np.float32),
        global_wn, base_wn)

    # ── Stage 2: 物理还原 + 叠加均用 float32 ────────────────────────────────
    print("[Stage 2] 替换物理还原+叠加 → float32 (A)...")
    profiles_S2, wn_grids_S2 = physical_restore_float32(
        pred_norm_Bparams, stats["y_mean"], stats["y_std"],
        S_B, nu0_B, delta0_B, base_wn)
    absorp_S2 = superposition_A(profiles_S2, wn_grids_S2, global_wn, base_wn)

    # ── Stage 3: 归一化+NN+物理还原+叠加全用 float32（但谱线参数仍为 B 的 float64）
    print("[Stage 3] 替换归一化+物理还原+叠加 → float32 (A)，谱线参数仍为 B...")
    inputs_raw_B = np.stack([gamma_d_B, gamma_l_B], axis=1).astype(np.float32)
    inputs_norm_f32 = ((inputs_raw_B - stats["x_mean"].astype(np.float32))
                       / stats["x_std"].astype(np.float32))
    tensor_in_f32 = torch.tensor(inputs_norm_f32, dtype=torch.float32)
    with torch.no_grad():
        pred_norm_f32 = model_B(tensor_in_f32).cpu().numpy()
    profiles_S3, wn_grids_S3 = physical_restore_float32(
        pred_norm_f32, stats["y_mean"], stats["y_std"],
        S_B, nu0_B, delta0_B, base_wn)
    absorp_S3 = superposition_A(profiles_S3, wn_grids_S3, global_wn, base_wn)

    # ── Stage 4: 谱线参数也改为 A 的 float32（全部 A 实现）──────────────────
    print("[Stage 4] 全部替换为 A 的 float32（含谱线参数）...")
    inputs_raw_A = np.stack([gamma_d_A, gamma_l_A], axis=1).astype(np.float32)
    inputs_norm_A = ((inputs_raw_A - stats["x_mean"].astype(np.float32))
                     / stats["x_std"].astype(np.float32))
    tensor_in_A = torch.tensor(inputs_norm_A, dtype=torch.float32)
    with torch.no_grad():
        pred_norm_A = model_B(tensor_in_A).cpu().numpy()
    profiles_S4, wn_grids_S4 = physical_restore_float32(
        pred_norm_A, stats["y_mean"], stats["y_std"],
        S_A, nu0_A, delta0_A, base_wn)
    absorp_S4 = superposition_A(profiles_S4, wn_grids_S4, global_wn, base_wn)

    # ── 打印统计 ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("逐阶段误差消融（以 Stage 0 / B 完整 float64 流水线为参考）")
    print("="*70)
    for label, absorp in [
        ("Stage 1: 仅替换叠加→float32",              absorp_S1),
        ("Stage 2: +物理还原→float32",               absorp_S2),
        ("Stage 3: +输入归一化→float32 (谱参数=B)",  absorp_S3),
        ("Stage 4: +谱线参数→float32  (=流水线A)",   absorp_S4),
    ]:
        err = relative_error_pct(absorp, absorp_B)
        print(f"  {label}")
        print(f"    {stats_str(err)}")
    print("="*70)

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("O2 12975–13150 cm⁻¹ | P=2000 Pa T=220 K\n误差逐阶段消融（参考=B float64全程）",
                 fontsize=11)

    stage_data = [
        ("Stage 1: 仅叠加→float32",                  absorp_S1),
        ("Stage 2: +物理还原→float32",               absorp_S2),
        ("Stage 3: +归一化→float32 (谱参数=B)",      absorp_S3),
        ("Stage 4: +谱线参数→float32 (=流水线A)",    absorp_S4),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for ax, (label, absorp), color in zip(axes, stage_data, colors):
        err = relative_error_pct(absorp, absorp_B)
        ax.plot(global_wn, err, lw=0.5, color=color)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel("Relative error [%]")
        ax.set_title(f"{label}   {stats_str(err)}", fontsize=9)

    axes[-1].set_xlabel("Wavenumber [cm⁻¹]")
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n消融分析图已保存: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
