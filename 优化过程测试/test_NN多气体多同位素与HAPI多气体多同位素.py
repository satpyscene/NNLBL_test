import h5py
import numpy as np
import matplotlib.pyplot as plt


def compare_hapi_outputs(file_path):
    # 1. 读取数据
    with h5py.File(file_path, "r") as f_fold:
        # 假设数据路径为 model_output/layer_000
        # 如果 layer_000 下还有子字段（如 'wavenumber' 和 'absorption'），请根据实际修改
        NN_data = f_fold["model_output/layer_000"][:]
        HAPI_data = f_fold["hapi_benchmark/layer_000"][:]

        # 如果文件中存储了波数轴，也一并读取（假设两文件波数轴一致）
        # wavenumbers = f_old['model_output/wavenumbers'][:]
        # 如果没有，则生成索引轴
        x_axis = np.arange(len(NN_data))

    # 2. 计算差异
    diff = NN_data - HAPI_data
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(diff)
    RE = diff / np.max(HAPI_data)
    print(f"对比结果:")
    print(f"最大绝对偏差: {max_diff:.30e}")
    print(f"平均偏差: {mean_diff:.30e}")

    # 3. 绘图可视化
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # 上图：两条曲线叠加
    ax1.plot(x_axis, HAPI_data, label="Old (PYTIPS)", alpha=0.8)
    ax1.plot(x_axis, NN_data, label="New (partitionSum)", linestyle="--", alpha=0.8)
    ax1.set_ylabel("Value (Absorption / Transmission)")
    ax1.set_yscale("log")
    ax1.set_title("Comparison of CO2 Model Output (Layer 000)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：残差图
    ax2.plot(x_axis, RE * 100, color="red", label="Residual (New - Old)/Old")
    ax2.set_ylabel("RE %")
    ax2.set_xlabel("Index / Wavenumber")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 使用示例
compare_hapi_outputs(
    # "sigma_output_filefold/CO2_600_700_0.01_101325_296_NN和HAPI均多同位素适配.h5",
    "sigma_output_filefold/H2O_Iso1-2_CO2_Iso1-2-3_O3_Iso1-2_600_700_0.01_101325_296.h5",
)
