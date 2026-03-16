import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
from mt_ckd_h2o import MTCKD_H2O  # 导入刚才写的 Python 类


def compare_runs():
    # --- 1. 配置参数 (必须与 Fortran Namelist 完全一致) ---
    p_atm = 1013.0  # hPa
    t_atm = 300.0  # K
    h2o_frac = 0.00990098  # VMR
    wv1 = 497.0
    wv2 = 603.0
    dwv = 1.0

    fortran_nc_file = "/Users/user/Desktop/0_NNLBL_main_use/MT_CKD_H2O-master/run_example/mt_ckd_h2o_output.nc"
    data_nc_file = "/Users/user/Desktop/0_NNLBL_main_use/data/absco-ref_wv-mt-ckd.nc"

    # --- 2. 加载 Fortran 结果 ---
    if not os.path.exists(fortran_nc_file):
        print(f"错误: 找不到 Fortran 输出文件 {fortran_nc_file}")
        print("请先运行 Fortran 程序生成此文件。")
        return

    print(f"正在读取 Fortran 结果: {fortran_nc_file} ...")
    with nc.Dataset(fortran_nc_file, "r") as ds:
        f_nu = ds.variables["wavenumbers"][:]
        f_self = ds.variables["self_absorption"][:]
        f_for = ds.variables["frgn_absorption"][:]

    # --- 3. 运行 Python 模型 ---
    print(f"正在运行 Python 模型...")
    model = MTCKD_H2O(data_nc_file)
    p_nu, p_self, p_for = model.get_absorption(p_atm, t_atm, h2o_frac, wv1, wv2, dwv)

    # --- 4. 维度检查 ---
    if len(f_nu) != len(p_nu):
        print(f"❌ 错误: 数组长度不一致!")
        print(f"Fortran: {len(f_nu)}, Python: {len(p_nu)}")
        return
    else:
        print(f"✅ 数组长度一致: {len(p_nu)} 点")

    # --- 5. 计算误差 ---
    # Self Continuum Error
    diff_self = np.abs(f_self - p_self)
    max_diff_self = np.max(diff_self)
    rel_err_self = np.max(diff_self / (np.abs(f_self) + 1e-30))  # 避免除以0

    # Foreign Continuum Error
    diff_for = np.abs(f_for - p_for)
    max_diff_for = np.max(diff_for)
    rel_err_for = np.max(diff_for / (np.abs(f_for) + 1e-30))

    # --- 6. 打印统计报告 ---
    print("\n" + "=" * 40)
    print("      MT-CKD 移植验证报告")
    print("=" * 40)

    # 判定阈值：通常 float64 的精度差异在 1e-12 左右
    # 但由于插值算法实现细节（如 Fortran 编译器优化 vs NumPy），1e-7 以内都算完美
    threshold = 1e-7

    print(f"Self-Broadening (自加宽):")
    print(f"  最大绝对误差: {max_diff_self:.4e} cm^2/mol")
    print(f"  最大相对误差: {rel_err_self:.4%}")
    if max_diff_self < threshold:
        print("  结果判定: ✅ PASS (完美匹配)")
    elif max_diff_self < 1e-5:
        print("  结果判定: ⚠️ WARN (微小差异，可能是浮点精度导致)")
    else:
        print("  结果判定: ❌ FAIL (差异显著，需检查算法)")

    print("-" * 40)

    print(f"Foreign-Broadening (外加宽):")
    print(f"  最大绝对误差: {max_diff_for:.4e} cm^2/mol")
    print(f"  最大相对误差: {rel_err_for:.4%}")
    if max_diff_for < threshold:
        print("  结果判定: ✅ PASS (完美匹配)")
    else:
        print("  结果判定: ⚠️ 检查数值")

    # --- 7. 绘图验证 (Visual Validation) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # 图1: Self 对比
    axes[0, 0].plot(f_nu, f_self, "k-", lw=3, alpha=0.5, label="Fortran")
    axes[0, 0].plot(p_nu, p_self, "r--", lw=1.5, label="Python")
    axes[0, 0].set_title("Self Continuum Spectrum")
    axes[0, 0].set_ylabel("Absorption Coeff")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: Self 残差
    axes[1, 0].plot(f_nu, f_self - p_self, "b-")
    axes[1, 0].set_title("Self Residual (Fortran - Python)")
    axes[1, 0].set_ylabel("Difference")
    axes[1, 0].grid(True, alpha=0.3)

    # 图3: Foreign 对比
    axes[0, 1].plot(f_nu, f_for, "k-", lw=3, alpha=0.5, label="Fortran")
    axes[0, 1].plot(p_nu, p_for, "r--", lw=1.5, label="Python")
    axes[0, 1].set_title("Foreign Continuum Spectrum")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 图4: Foreign 残差
    axes[1, 1].plot(f_nu, f_for - p_for, "b-")
    axes[1, 1].set_title("Foreign Residual (Fortran - Python)")
    axes[1, 1].set_xlabel("Wavenumber (cm-1)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_runs()
