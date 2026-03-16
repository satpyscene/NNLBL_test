import h5py
import numpy as np
import matplotlib.pyplot as plt


def compare_hapi_outputs_three(file_paths, labels):
    # 1. 读取数据
    data_list = []
    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            data = f["model_output/layer_000"][:]
            data_list.append(data)

    # 假设三个文件长度一致
    # x_axis = np.arange(len(data_list[0]))
    x_axis = np.arange(4800, 5200.001, 0.01)
    # 2. 以第一个文件作为基准，计算差异和相对误差
    base_data = data_list[0]
    diffs = [data - base_data for data in data_list[1:]]
    res = [diff / base_data for diff in diffs]

    print("对比结果（相对于基准 vmr）:")
    for i, diff in enumerate(diffs):
        print(f"{labels[i+1]} vs {labels[0]}")
        print(f"  最大绝对偏差: {np.max(np.abs(diff)):.30e}")
        print(f"  平均偏差: {np.mean(diff):.30e}")

    # 3. 绘图可视化
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        sharex=True,
        # gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
    )

    # 上图：三条吸收截面曲线
    for data, label in zip(data_list, labels):
        axes.plot(x_axis, data, label=label, alpha=0.8)
    axes.set_yscale("log")
    axes.set_ylabel("Absorption Cross Section (cm$^2$/molc.)")
    axes.set_xlabel("Wavenumber cm$^{-1}$")
    axes.set_title("Impact of Different H2O VMR Settings on Absorption Cross Section")
    axes.legend()
    axes.grid(True, alpha=0.3)

    # # 中图：绝对差异
    # for diff, label in zip(diffs, labels[1:]):
    #     axes[1].plot(x_axis, diff, label=f"{label} - {labels[0]}")
    # axes[1].set_ylabel("Absolute Difference")
    # axes[1].grid(True, alpha=0.3)
    # axes[1].legend()

    # # 下图：相对误差（百分比）
    # for re, label in zip(res, labels[1:]):
    #     axes[2].plot(x_axis, re * 100, label=f"{label} vs {labels[0]}")
    # axes[2].set_ylabel("Relative Error (%)")
    # axes[2].set_xlabel("Index / Wavenumber")
    # axes[2].grid(True, alpha=0.3)
    # axes[2].legend()

    plt.tight_layout()
    plt.show()


# 使用示例
compare_hapi_outputs_three(
    file_paths=[
        "../sigma_output_filefold/vmr0p0001_H2O_Major_4800_5200_0.01_101325_296.h5",
        "../sigma_output_filefold/vmr0p01_H2O_Major_4800_5200_0.01_101325_296.h5",
        "../sigma_output_filefold/vmr0p04_H2O_Major_4800_5200_0.01_101325_296.h5",
    ],
    labels=[
        "H2O vmr = 0.0001 (baseline)",
        "H2O vmr = 0.01",
        "H2O vmr = 0.04",
    ],
)
