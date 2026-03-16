import h5py
import numpy as np
import matplotlib.pyplot as plt


def compare_hapi_outputs(old_file_path, new_file_path):
    # 1. 读取数据
    with h5py.File(old_file_path, "r") as f_old, h5py.File(new_file_path, "r") as f_new:
        # 假设数据路径为 model_output/layer_000
        # 如果 layer_000 下还有子字段（如 'wavenumber' 和 'absorption'），请根据实际修改

        old_data = f_old["model_output/layer_000"][:]
        new_data = f_new["model_output/layer_000"][:]

        # old_data = f_old["hapi_benchmark/layer_000"][:]
        # new_data = f_new["hapi_benchmark/layer_000"][:]

        # old_data = f_new["hapi_benchmark/layer_000"][:]
        # new_data = f_new["model_output/layer_000"][:]

        # 如果文件中存储了波数轴，也一并读取（假设两文件波数轴一致）
        # wavenumbers = f_old['model_output/wavenumbers'][:]
        # 如果没有，则生成索引轴
        # x_axis = np.arange(len(old_data))
        x_axis = np.arange(4800, 5200.001, 0.01)
    # 2. 计算差异
    diff = new_data - old_data
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(diff)
    RE = diff / old_data
    print(f"对比结果:")
    print(f"最大绝对偏差: {max_diff:.30e}")
    print(f"平均偏差: {mean_diff:.30e}")

    # 3. 绘图可视化
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

    # 上图：两条曲线叠加
    ax1.plot(x_axis, old_data, label="no continuum", alpha=0.8)
    ax1.plot(x_axis, new_data, label="continuum", linestyle="--", alpha=0.8)
    ax1.set_ylabel("Absorption Cross Section (cm$^2$/molc.)")
    ax1.set_xlabel("Wavenumber cm$^{-1}$")
    ax1.set_yscale("log")
    ax1.set_title("Impact of continnum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # # 下图：残差图
    # ax2.plot(x_axis, RE * 100, color="red", label="Residual (New - Old)/Old")
    # ax2.set_ylabel("RE %")
    # ax2.set_xlabel("Index / Wavenumber")
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 使用示例
compare_hapi_outputs(
    "/Users/user/Desktop/0_NNLBL_main_use/sigma_output_filefold/H2O_Major_4800_5200_0.01_101325_296.h5",
    "/Users/user/Desktop/0_NNLBL_main_use/sigma_output_filefold/vmr0p04_H2O_Major_4800_5200_0.01_101325_296.h5",
)
