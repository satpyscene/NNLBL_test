"""
论文结果图生成脚本
用法：在项目根目录下运行
    python paper_results/generate_figures.py
"""
import sys
import os

# 将项目根目录加入路径，使 NNLBL_src 可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from NNLBL_src.single_level_K_plot import load_model, compute_absorption_for_condition
from paper_results.jqsrt_plot_utils import create_jqsrt_comparison_figure_relative_error_avg

# ============================================================
# 用户配置区域（按需修改）
# ============================================================

# -- 模型文件路径 --
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
HP_MODEL_PATH = os.path.join(PROJECT_ROOT, "NNmodel&stats/voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth")
HP_STATS_PATH = os.path.join(PROJECT_ROOT, "NNmodel&stats/voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy")
LP_MODEL_PATH = os.path.join(PROJECT_ROOT, "NNmodel&stats/voigt_model_best_lp_Full-nonuniform-n0_1000_noshift.pth")
LP_STATS_PATH = os.path.join(PROJECT_ROOT, "NNmodel&stats/lp_voigt_stats_Full-nonuniform-n0_1000_noshift.npy")

# -- 输出目录 --
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# -- 压强阈值（Pa）：低于此值使用低压模型，否则使用高压模型 --
PRESSURE_THRESHOLD_PA = 2200.0

# -- 模型超参数 --
INPUT_SIZE = 2
HIDDEN_SIZE = [100, 500, 1000, 500]
MODEL_OUTPUT_SIZE = 5001
WING_SIZE = 25.0
TOTAL_POINTS = 5001
CONCENTRATION_FACTOR = 6.0

# -- 测试条件 --
TEST_CASES = {
    "high_pressure": {
        "pressure_pa": 101395,
        "temperature_k": 279.1,
        "label": "High Pressure",
        "gases": [
            {"molecule": "O3",  "wn_min": 600,   "wn_max": 1200,  "wn_step": 0.01},
            {"molecule": "H2O", "wn_min": 4100,  "wn_max": 4400,  "wn_step": 0.01},
            {"molecule": "CO2", "wn_min": 2150,  "wn_max": 2400,  "wn_step": 0.01},
            {"molecule": "O2",  "wn_min": 12975, "wn_max": 13150, "wn_step": 0.01},
        ],
    },
    "low_pressure": {
        "pressure_pa": 2000,
        "temperature_k": 222.0,
        "label": "Low Pressure",
        "gases": [
            {"molecule": "O3",  "wn_min": 600,   "wn_max": 1200,  "wn_step": 0.01},
            {"molecule": "H2O", "wn_min": 4100,  "wn_max": 4400,  "wn_step": 0.01},
            {"molecule": "CO2", "wn_min": 2150,  "wn_max": 2400,  "wn_step": 0.01},
            {"molecule": "O2",  "wn_min": 12975, "wn_max": 13150, "wn_step": 0.01},
        ],
    },
}

# ============================================================
# 主程序（通常不需要修改）
# ============================================================

def main():
    device = torch.device("cpu")

    models = {
        "lp": load_model(LP_MODEL_PATH, INPUT_SIZE, HIDDEN_SIZE, MODEL_OUTPUT_SIZE, device),
        "hp": load_model(HP_MODEL_PATH, INPUT_SIZE, HIDDEN_SIZE, MODEL_OUTPUT_SIZE, device),
    }
    statistics_pool = {
        "lp": np.load(LP_STATS_PATH, allow_pickle=True).item(),
        "hp": np.load(HP_STATS_PATH, allow_pickle=True).item(),
    }

    for pressure_type, config in TEST_CASES.items():
        pressure_pa = config["pressure_pa"]
        temperature_k = config["temperature_k"]
        pressure_label = config["label"]
        model_key = "lp" if pressure_pa < PRESSURE_THRESHOLD_PA else "hp"

        results_list = []
        for gas in config["gases"]:
            print(f"[{pressure_label}] {gas['molecule']} | P={pressure_pa:.0f} Pa | T={temperature_k:.1f} K")
            result = compute_absorption_for_condition(
                molecule_name=gas["molecule"],
                wavenumber_min=gas["wn_min"],
                wavenumber_max=gas["wn_max"],
                wavenumber_step=gas["wn_step"],
                pressure_pa=pressure_pa,
                temperature_k=temperature_k,
                model=models[model_key],
                statistics=statistics_pool[model_key],
                device=device,
                wing_size=WING_SIZE,
                total_points=TOTAL_POINTS,
                concentration_factor=CONCENTRATION_FACTOR,
            )
            results_list.append(result)

        save_path = os.path.join(
            OUTPUT_DIR,
            f"JQSRT_{pressure_type}_P{int(pressure_pa)}_T{int(temperature_k)}.png",
        )
        create_jqsrt_comparison_figure_relative_error_avg(results_list, save_path)


if __name__ == "__main__":
    main()
