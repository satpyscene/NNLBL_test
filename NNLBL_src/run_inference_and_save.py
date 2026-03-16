"""
完整推理与数据保存脚本
功能：运行模型推理得到100层输出，计算HAPI基准，保存所有数据
"""

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
import torch
import torch.nn as nn
import numpy as np
import h5py
import time
import os
from tqdm import tqdm
from joblib import Parallel, delayed, Memory
from collections import OrderedDict
import matplotlib.pyplot as plt
from .hapi import (
    db_begin,
    fetch,
    fetch_by_ids,
    absorptionCoefficient_Voigt,
    calculateProfileParametersVoigt,
    tableList,
    PYTIPS,
)

# ============================================================================
# 配置
# ============================================================================

# 缓存配置
cachedir = "./.hapi_cache"
memory = Memory(cachedir, verbose=0)

# 模型路径
HP_MODEL_PATH = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/1_train/model_save/1000.0_nonuniform_5001-400-401_high_pressure_voigt_dataset_FULL_noshift_best_model.pth"
HP_STATS_PATH = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/1_train/stats_npy/1000.0_nonuniform_5001-400-401_high_pressure_voigt_dataset_FULL_noshift_stats.npy"
LP_MODEL_PATH = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/1_train/model_save/voigt_model_best_lp_Full-nonuniform-n0_1000_noshift.pth"
LP_STATS_PATH = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/1_train/stats_npy/lp_voigt_stats_Full-nonuniform-n0_1000_noshift.npy"

# 物理参数
PRESSURE_THRESHOLD_PA = 1000.0
MOLECULE = "CO2"
GLOBAL_WN_MIN = 1152
GLOBAL_WN_MAX = 1937
GLOBAL_WN_STEP = 0.01
WING_SIZE = 25.0
TOTAL_POINTS_FULL = 5001
CONCENTRATION_FACTOR = 6.0

# 美国标准大气廓线文件
PRESSURE_FILE = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/pres_100.txt"
TEMPERATURE_FILE = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/US_STANDARD_ATMOSPHERE_T.txt"
CO2_concentration_pmv = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/US_STANDARD_ATMOSPHERE_co2.txt"

# INDEX_FILE = "profile/representative_profiles_indices.txt"
# PROFILE_CHOICE = 0

# 输出路径
OUTPUT_H5 = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/3_100levels_precision&efficiency/output_data/100layers_sigma_NN&Hapi.h5"

# ============================================================================
# 神经网络模型
# ============================================================================


class SimpleNetWithPrePostProcess(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc5 = nn.Linear(hidden_size[3], output_size)

        self.register_buffer("x_mean", None)
        self.register_buffer("x_std", None)
        self.register_buffer("y_mean", None)
        self.register_buffer("y_std", None)

    def set_stats(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
        self.x_std = torch.tensor(x_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.tensor(y_std, dtype=torch.float32)

    def forward_with_full_pipeline(self, gamma_d, gamma_l, S, nu0, delta_0, base_grid):
        # 归一化在 float64 中进行（与 single_level_K_plot 的 numpy float64 归一化一致），
        # 再转回 float32 送入 NN（权重为 float32）
        inputs_raw = torch.stack([gamma_d, gamma_l], dim=1).to(torch.float64)
        normalized_inputs = ((inputs_raw - self.x_mean.to(torch.float64))
                             / self.x_std.to(torch.float64)).to(torch.float32)

        out = self.fc1(normalized_inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        predicted_normalized = self.fc5(out)

        # 物理还原升精度为 float64，消除 10^x 和 S 乘法的 float32 舍入误差
        pred_f64 = predicted_normalized.to(torch.float64)
        y_log_scale = pred_f64 * self.y_std.to(torch.float64) + self.y_mean.to(torch.float64)
        y_physical_scale = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=pred_f64.device), y_log_scale)
        final_profiles = y_physical_scale * S.to(torch.float64).unsqueeze(1)

        total_shift = (nu0.to(torch.float64) + delta_0.to(torch.float64)) - 1000.0
        final_wavenumber_grids = base_grid.to(torch.float64).unsqueeze(0) + total_shift.unsqueeze(1)

        return final_profiles, final_wavenumber_grids


def load_model(model_path, input_size, hidden_size, output_size, device, stats):
    model = SimpleNetWithPrePostProcess(input_size, hidden_size, output_size)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.set_stats(stats["x_mean"], stats["x_std"], stats["y_mean"], stats["y_std"])
    model.to(device)
    model.eval()
    return model


# ============================================================================
# HAPI物理参数计算
# ============================================================================


# @memory.cache
# def get_hapi_physical_params(
#     molecule_name, wavenumber_min, wavenumber_max, temperature_k, pressure_pa
# ):
#     MOLECULE_IDS = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
#     molecule_id = MOLECULE_IDS[molecule_name]
#     pressure_atm = pressure_pa / 101325.0
#     db_path = f"data/{molecule_name}_hapi"

#     # --- 新增：检查路径是否存在，不存在则创建 ---
#     if not os.path.exists(db_path):
#         os.makedirs(db_path, exist_ok=True)
#         print(f"已创建HAPI数据库目录: {db_path}")
#     # ----------------------------------------
#     db_begin(db_path)
#     table_name = f"{molecule_name}_{wavenumber_min}_{wavenumber_max}"

#     if table_name not in tableList():
#         fetch(table_name, molecule_id, 1, wavenumber_min, wavenumber_max)

#     try:
#         from .hapi import LOCAL_TABLE_CACHE

#         DATA_DICT = LOCAL_TABLE_CACHE[table_name]["data"]
#         num_total_lines = len(DATA_DICT["nu"])
#     except (KeyError, TypeError):
#         return {
#             k: np.array([], dtype=np.float32)
#             for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]
#         }

#     if num_total_lines == 0:
#         return {
#             k: np.array([], dtype=np.float32)
#             for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]
#         }

#     lines_params = {
#         k: np.empty(num_total_lines, dtype=np.float32)
#         for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]
#     }

#     for i in range(num_total_lines):
#         trans = {param: DATA_DICT[param][i] for param in DATA_DICT}
#         trans["T"], trans["p"] = temperature_k, pressure_atm
#         trans["T_ref"], trans["p_ref"] = 296.0, 1.0
#         trans["Diluent"] = {"air": 1.0}
#         trans["SigmaT"] = PYTIPS(trans["molec_id"], trans["local_iso_id"], trans["T"])
#         trans["SigmaT_ref"] = PYTIPS(
#             trans["molec_id"], trans["local_iso_id"], trans["T_ref"]
#         )

#         line_calc_params = calculateProfileParametersVoigt(TRANS=trans)

#         lines_params["gamma_d"][i] = line_calc_params["GammaD"]
#         lines_params["gamma_l"][i] = line_calc_params["Gamma0"]
#         lines_params["delta_0"][i] = line_calc_params["Delta0"]
#         lines_params["nu0"][i] = trans["nu"]
#         lines_params["S"][i] = line_calc_params["Sw"]

#     return lines_params


@memory.cache
def get_hapi_physical_params_new(
    molecule_name,
    wavenumber_min,
    wavenumber_max,
    temperature_k,
    pressure_pa,
    global_iso_ids=None,  # 接收全局 ID 列表，例如 [1, 2] 代表水汽的 H216O 和 H218O
    vmr=0.0,
):
    # 1. 基础参数准备
    pressure_atm = pressure_pa / 101325.0
    db_path = f"data/{molecule_name}_hapi"

    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)

    db_begin(db_path)

    # 2. 表名与数据下载
    # 使用 global_iso_ids 构建唯一表名，防止不同同位素组合冲突
    iso_tag = "_".join(map(str, global_iso_ids)) if global_iso_ids else "default"
    table_name = f"{molecule_name}_{wavenumber_min}_{wavenumber_max}_{iso_tag}"

    if table_name not in tableList():
        if global_iso_ids:
            # 使用你提供的多同位素下载函数
            fetch_by_ids(table_name, global_iso_ids, wavenumber_min, wavenumber_max)
        else:
            # 兼容逻辑：如果没有指定，默认下载该分子的主同位素
            # 这里假定 MOLECULE_IDS 映射依然存在
            MOLECULE_IDS = {
                "H2O": 1,
                "CO2": 2,
                "O3": 3,
                "N2O": 4,
                "CO": 5,
                "CH4": 6,
                "O2": 7,
            }
            fetch(
                table_name,
                MOLECULE_IDS[molecule_name],
                1,
                wavenumber_min,
                wavenumber_max,
            )

    # 3. 数据提取与空检查
    try:
        from .hapi import LOCAL_TABLE_CACHE

        DATA_DICT = LOCAL_TABLE_CACHE[table_name]["data"]
        num_total_lines = len(DATA_DICT["nu"])
    except (KeyError, TypeError):
        return {
            k: np.array([], dtype=np.float32)
            for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]
        }

    if num_total_lines == 0:
        return {
            k: np.array([], dtype=np.float32)
            for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]
        }

    # 4. 预分配内存
    # nu0 和 delta_0 必须 float64：非均匀网格中心间距 ~3e-4 cm⁻¹，
    # float32 精度 ~1e-3 cm⁻¹ 会导致 wn_grid 偏移 ~1-2 个格点，引入大误差
    lines_params = {
        k: np.zeros(num_total_lines, dtype=np.float64 if k in ("nu0", "delta_0") else np.float32)
        for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]
    }

    # 5. 核心循环：计算物理参数
    # 提前取出常用列以提高循环效率
    molec_ids = DATA_DICT["molec_id"]
    local_iso_ids = DATA_DICT["local_iso_id"]
    nus = DATA_DICT["nu"]
    print("debug:molec_ids[0]=", molec_ids[0])
    is_h2o_data = len(molec_ids) > 0 and int(molec_ids[0]) == 1

    # 自加宽修正逻辑：
    # self = 当前水汽浓度
    # air = 剩下的背景气体
    current_diluent = {"self": vmr, "air": 1.0 - vmr}
    print(current_diluent, "分子自加宽订正gammaL生效")

    for i in range(num_total_lines):
        # 构建当前谱线的转换字典
        trans = {param: DATA_DICT[param][i] for param in DATA_DICT}
        trans["T"], trans["p"] = temperature_k, pressure_atm
        trans["T_ref"], trans["p_ref"] = 296.0, 1.0
        trans["Diluent"] = current_diluent

        # --- 优雅的配分函数处理 ---
        # 直接使用 DATA_DICT 中自带的分子和局部同位素 ID
        m = molec_ids[i]
        iso = local_iso_ids[i]

        # 使用与 absorptionCoefficient_Voigt 内部一致的配分函数 PYTIPS
        trans["SigmaT"] = PYTIPS(m, iso, temperature_k)
        trans["SigmaT_ref"] = PYTIPS(m, iso, 296.0)

        # 计算 Voigt 参数
        line_calc_params = calculateProfileParametersVoigt(TRANS=trans)

        lines_params["gamma_d"][i] = line_calc_params["GammaD"]
        lines_params["gamma_l"][i] = line_calc_params["Gamma0"]
        lines_params["delta_0"][i] = line_calc_params["Delta0"]
        lines_params["nu0"][i] = nus[i]
        lines_params["S"][i] = line_calc_params["Sw"]

    return lines_params


# ============================================================================
# 网格与插值
# ============================================================================


def create_non_uniform_grid(nu0, wing_cm, total_points, concentration_factor):
    x_uniform = np.linspace(-1, 1, total_points)
    x_non_uniform = np.sinh(x_uniform * concentration_factor) / np.sinh(
        concentration_factor
    )
    return nu0 + wing_cm * x_non_uniform


def pack_layers_into_batch(
    layer_indices,
    all_layers_lines_params,
    DEVICE,
    gamma_l_threshold=None,  # 新增：洛伦兹半高半宽阈值
    use_high_gamma=True      # 新增：True表示使用>=阈值的谱线(HP)，False表示使用<阈值的谱线(LP)
):
    """
    将多层的谱线参数打包成batch用于GPU推理

    新增功能：支持根据gamma_l阈值过滤谱线
    - 如果gamma_l_threshold不为None，则根据use_high_gamma选择性打包谱线
    - use_high_gamma=True: 只打包gamma_l >= threshold的谱线（用于HP模型）
    - use_high_gamma=False: 只打包gamma_l < threshold的谱线（用于LP模型）
    """
    valid_indices = [
        i for i in layer_indices if len(all_layers_lines_params[i]["gamma_d"]) > 0
    ]

    if not valid_indices:
        empty = torch.empty(0, dtype=torch.float32, device=DEVICE)
        return {k: empty for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]}, [
            0
        ] * (len(layer_indices) + 1)

    # 收集数据时应用gamma_l过滤
    all_gamma_d_list = []
    all_gamma_l_list = []
    all_S_list = []
    all_nu0_list = []
    all_delta_0_list = []
    layer_boundaries = [0]

    for i in layer_indices:
        params = all_layers_lines_params[i]

        # 如果指定了gamma_l阈值，进行过滤
        if gamma_l_threshold is not None:
            gamma_l_arr = params["gamma_l"]
            if use_high_gamma:
                # 选择gamma_l >= threshold的谱线（高压/宽谱线）
                mask = gamma_l_arr >= gamma_l_threshold
            else:
                # 选择gamma_l < threshold的谱线（低压/窄谱线）
                mask = gamma_l_arr < gamma_l_threshold

            # 应用mask过滤所有参数
            filtered_gamma_d = params["gamma_d"][mask]
            filtered_gamma_l = params["gamma_l"][mask]
            filtered_S = params["S"][mask]
            filtered_nu0 = params["nu0"][mask]
            filtered_delta_0 = params["delta_0"][mask]
        else:
            # 如果没有指定阈值，使用所有谱线（保持原有行为）
            filtered_gamma_d = params["gamma_d"]
            filtered_gamma_l = params["gamma_l"]
            filtered_S = params["S"]
            filtered_nu0 = params["nu0"]
            filtered_delta_0 = params["delta_0"]

        # 添加到列表
        all_gamma_d_list.append(filtered_gamma_d)
        all_gamma_l_list.append(filtered_gamma_l)
        all_S_list.append(filtered_S)
        all_nu0_list.append(filtered_nu0)
        all_delta_0_list.append(filtered_delta_0)

        # 更新边界
        layer_boundaries.append(
            layer_boundaries[-1] + len(filtered_gamma_d)
        )

    # 合并所有层的数据
    all_gamma_d = np.concatenate(all_gamma_d_list) if all_gamma_d_list else np.array([])
    all_gamma_l = np.concatenate(all_gamma_l_list) if all_gamma_l_list else np.array([])
    all_S = np.concatenate(all_S_list) if all_S_list else np.array([])
    all_nu0 = np.concatenate(all_nu0_list) if all_nu0_list else np.array([])
    all_delta_0 = np.concatenate(all_delta_0_list) if all_delta_0_list else np.array([])

    # 如果过滤后没有数据，返回空结果
    if len(all_gamma_d) == 0:
        empty = torch.empty(0, dtype=torch.float32, device=DEVICE)
        return {k: empty for k in ["gamma_d", "gamma_l", "S", "nu0", "delta_0"]}, [
            0
        ] * (len(layer_indices) + 1)

    batch_tensors = {
        "gamma_d": torch.from_numpy(all_gamma_d).to(DEVICE),
        "gamma_l": torch.from_numpy(all_gamma_l).to(DEVICE),
        "S": torch.from_numpy(all_S).to(DEVICE),
        "nu0": torch.from_numpy(all_nu0).to(DEVICE),
        "delta_0": torch.from_numpy(all_delta_0).to(DEVICE),
    }

    return batch_tensors, layer_boundaries


def process_mega_batch_gpu(
    layer_indices,
    all_layers_lines_params,
    model,
    base_grid_gpu,
    DEVICE,
    gamma_l_threshold=None,  # 新增：洛伦兹半高半宽阈值
    use_high_gamma=True      # 新增：选择高/低gamma_l的谱线
):
    batch_tensors, layer_boundaries = pack_layers_into_batch(
        layer_indices,
        all_layers_lines_params,
        DEVICE,
        gamma_l_threshold=gamma_l_threshold,
        use_high_gamma=use_high_gamma
    )

    if batch_tensors["gamma_d"].numel() == 0:
        return [(None, None) for _ in layer_indices]

    with torch.no_grad():
        all_profiles_gpu, all_wn_grids_gpu = model.forward_with_full_pipeline(
            batch_tensors["gamma_d"],
            batch_tensors["gamma_l"],
            batch_tensors["S"],
            batch_tensors["nu0"],
            batch_tensors["delta_0"],
            base_grid_gpu,
        )

    layer_gpu_results = []
    for i in range(len(layer_indices)):
        start_idx, end_idx = layer_boundaries[i], layer_boundaries[i + 1]
        if start_idx == end_idx:
            layer_gpu_results.append((None, None))
        else:
            layer_gpu_results.append(
                (
                    all_profiles_gpu[start_idx:end_idx],
                    all_wn_grids_gpu[start_idx:end_idx],
                )
            )

    return layer_gpu_results


def perform_superposition_gpu(
    final_profiles_gpu,
    final_wn_grids_gpu,
    global_wavenumber_grid,
    base_wavenumber_grid,
    batch_size=500,
):
    device = final_profiles_gpu.device
    num_lines = final_profiles_gpu.shape[0]

    global_grid_gpu = (
        torch.from_numpy(global_wavenumber_grid).to(device).to(torch.float64)
    )
    base_grid_gpu = torch.from_numpy(base_wavenumber_grid).to(device).to(torch.float64)
    absorption_gpu = torch.zeros(
        len(global_wavenumber_grid), dtype=torch.float64, device=device
    )

    final_profiles_gpu = final_profiles_gpu.to(torch.float64)
    final_wn_grids_gpu = final_wn_grids_gpu.to(torch.float64)
    shifts_gpu = final_wn_grids_gpu[:, 0] - base_grid_gpu[0]

    for batch_idx in range((num_lines + batch_size - 1) // batch_size):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_lines)

        batch_shifts = shifts_gpu[batch_start:batch_end]
        batch_profiles = final_profiles_gpu[batch_start:batch_end]

        batch_min = base_grid_gpu[0] + batch_shifts
        batch_max = base_grid_gpu[-1] + batch_shifts

        coverage_mask = (global_grid_gpu.unsqueeze(0) >= batch_min.unsqueeze(1)) & (
            global_grid_gpu.unsqueeze(0) <= batch_max.unsqueeze(1)
        )

        line_indices, global_indices = torch.where(coverage_mask)
        if line_indices.numel() == 0:
            continue

        rel_positions = global_grid_gpu[global_indices] - batch_shifts[line_indices]
        indices = torch.searchsorted(base_grid_gpu, rel_positions)
        valid_mask = (indices > 0) & (indices < len(base_grid_gpu))

        line_indices = line_indices[valid_mask]
        global_indices = global_indices[valid_mask]
        indices = indices[valid_mask]
        rel_positions = rel_positions[valid_mask]

        if line_indices.numel() == 0:
            continue

        x1 = base_grid_gpu[indices - 1]
        x2 = base_grid_gpu[indices]
        y1 = batch_profiles[line_indices, indices - 1]
        y2 = batch_profiles[line_indices, indices]

        weights = (rel_positions - x1) / (x2 - x1)
        interpolated = (y1 * (1 - weights) + y2 * weights)

        absorption_gpu.index_add_(0, global_indices, interpolated)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return absorption_gpu.cpu().numpy()


def process_superposition_from_gpu(profiles_gpu, wn_grids_gpu, global_grid, base_grid):
    if profiles_gpu is None:
        return np.zeros_like(global_grid)
    return perform_superposition_gpu(
        profiles_gpu, wn_grids_gpu, global_grid, base_grid, batch_size=500
    )


# ============================================================================
# HAPI基准计算
# ============================================================================


# def calculate_hapi_benchmark(
#     molecule_name,
#     wavenumber_grid,
#     temperature_k,
#     pressure_pa,
#     wavenumber_min,
#     wavenumber_max,
# ):
#     MOLECULE_IDS = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
#     molecule_id = MOLECULE_IDS[molecule_name]
#     pressure_atm = pressure_pa / 101325.0

#     db_path = f"data/{molecule_name}_hapi"
#     db_begin(db_path)
#     table_name = f"{molecule_name}_{wavenumber_min}_{wavenumber_max}"

#     if table_name not in tableList():
#         fetch(table_name, molecule_id, 1, wavenumber_min, wavenumber_max)

#     nu, coef = absorptionCoefficient_Voigt(
#         SourceTables=table_name,
#         WavenumberGrid=wavenumber_grid,
#         Environment={"p": pressure_atm, "T": temperature_k},
#         HITRAN_units=True,
#         WavenumberWing=25,
#     )

#     if coef is None:
#         coef = np.zeros_like(wavenumber_grid)
#     return coef


def calculate_hapi_benchmark_new(
    molecule_name,
    wavenumber_grid,
    temperature_k,
    pressure_pa,
    wavenumber_min,
    wavenumber_max,
    global_iso_ids=None,  # 新增参数
    vmr=0.0,
):
    # 1. 参数预处理
    pressure_atm = pressure_pa / 101325.0
    db_path = f"data/{molecule_name}_hapi"

    # 2. 数据库连接与表名构建
    db_begin(db_path)

    # 为了防止缓存冲突，表名需要包含同位素后缀
    iso_tag = "_".join(map(str, global_iso_ids)) if global_iso_ids else "1"
    table_name = f"{molecule_name}_{wavenumber_min}_{wavenumber_max}_{iso_tag}"

    # 3. 数据下载逻辑（适配多同位素）
    if table_name not in tableList():
        if global_iso_ids:
            # 使用全局 ID 下载多同位素数据
            fetch_by_ids(table_name, global_iso_ids, wavenumber_min, wavenumber_max)
        else:
            # 默认兼容逻辑：下载该分子的主同位素
            MOLECULE_IDS = {
                "H2O": 1,
                "CO2": 2,
                "O3": 3,
                "N2O": 4,
                "CO": 5,
                "CH4": 6,
                "O2": 7,
            }
            molecule_id = MOLECULE_IDS[molecule_name]
            fetch(table_name, molecule_id, 1, wavenumber_min, wavenumber_max)

    diluent_settings = {"self": vmr, "air": 1.0 - vmr}

    # 4. 调用 HAPI 原生函数计算吸收截面
    # HAPI 在计算时会自动根据表中的 molec_id 和 local_iso_id 处理各自的 TIPS 配分函数
    nu, coef = absorptionCoefficient_Voigt(
        SourceTables=table_name,
        WavenumberGrid=wavenumber_grid,
        Environment={"p": pressure_atm, "T": temperature_k},
        Diluent=diluent_settings,
        HITRAN_units=True,
        WavenumberWing=25,
    )

    # 5. 空结果处理
    if coef is None or len(coef) == 0:
        coef = np.zeros_like(wavenumber_grid, dtype=np.float32)

    return coef


# ============================================================================
# 保存HDF5
# ============================================================================


def save_to_hdf5(
    output_path,
    atmospheric_profile,
    model_results,
    hapi_results,
    wavenumber_grid,
    molecule_name,
):
    # print(f"\n正在保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.attrs["molecule"] = molecule_name
        f.attrs["num_layers"] = len(atmospheric_profile)
        f.attrs["wavenumber_min"] = wavenumber_grid.min()
        f.attrs["wavenumber_max"] = wavenumber_grid.max()
        f.attrs["has_hapi_data"] = hapi_results is not None

        f.create_dataset("wavenumber_grid", data=wavenumber_grid, compression="gzip")

        profile_group = f.create_group("atmospheric_profile")
        profile_group.create_dataset(
            "pressure_pa",
            data=np.array([l["pressure_pa"] for l in atmospheric_profile]),
            compression="gzip",
        )
        profile_group.create_dataset(
            "temperature_k",
            data=np.array([l["temperature_k"] for l in atmospheric_profile]),
            compression="gzip",
        )
        # profile_group.create_dataset(
        #     "CO2_ppmv",
        #     data=np.array([l["CO2_ppmv"] for l in atmospheric_profile]),
        #     compression="gzip",
        # )
        model_group = f.create_group("model_output")
        for i, data in enumerate(model_results):
            model_group.create_dataset(f"layer_{i:03d}", data=data, compression="gzip")

        if hapi_results is not None:
            hapi_group = f.create_group("hapi_benchmark")
            for i, data in enumerate(hapi_results):
                hapi_group.create_dataset(
                    f"layer_{i:03d}", data=data, compression="gzip"
                )

    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"NNLBL输出吸收截面光谱保存至{output_path}! 文件大小: {file_size:.2f} MB\n")


# # ============================================================================
# # 主程序
# # ============================================================================

# if __name__ == "__main__":

#     import sys

#     global_iso_ids = (None,)
#     # 命令行参数: --skip-hapi 跳过HAPI计算
#     SKIP_HAPI = "--skip-hapi" in sys.argv

#     print("=" * 80)
#     print("完整推理与数据保存")
#     if SKIP_HAPI:
#         print("⚠️  跳过HAPI计算模式")
#     print("=" * 80)

#     # 设置设备
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {DEVICE}")

#     # 全局网格
#     global_wavenumber_grid = np.arange(GLOBAL_WN_MIN, GLOBAL_WN_MAX, GLOBAL_WN_STEP)
#     base_wavenumber_grid = create_non_uniform_grid(
#         1000.0, WING_SIZE, TOTAL_POINTS_FULL, CONCENTRATION_FACTOR
#     )
#     base_grid_gpu = torch.tensor(
#         base_wavenumber_grid, dtype=torch.float32, device=DEVICE
#     )

#     # 加载大气廓线
#     print("\n加载大气廓线...")
#     pressures_pa = np.loadtxt(PRESSURE_FILE) * 100
#     # representative_indices = np.loadtxt(INDEX_FILE, dtype=int)
#     # target_column = representative_indices[PROFILE_CHOICE] - 1
#     temperatures_k = np.loadtxt(TEMPERATURE_FILE)
#     CO2_concentration = np.loadtxt(CO2_concentration_pmv)

#     # 1. 处理 Level (边界)：直接存储原始的 101 个值
#     atmospheric_levels = [
#         {"level": i, "pressure_pa": p, "temperature_k": t, "CO2_ppmv": c}
#         for i, (p, t, c) in enumerate(
#             zip(pressures_pa, temperatures_k, CO2_concentration)
#         )
#     ]

#     # 2. 处理 Layer (层)：计算相邻边界的平均值，共 100 层
#     # 遍历范围是 0 到 N-2 (即 len - 1)
#     atmospheric_profile = [
#         {
#             "layer": i,
#             "pressure_pa": (pressures_pa[i] + pressures_pa[i + 1]) / 2,  # 气压平均
#             "temperature_k": (temperatures_k[i] + temperatures_k[i + 1])
#             / 2,  # 温度平均
#             "CO2_ppmv": (CO2_concentration[i] + CO2_concentration[i + 1])
#             / 2,  # CO2浓度平均
#         }
#         for i in range(len(pressures_pa) - 1)
#     ]

#     # 打印结果验证
#     print(f"✅ 成功加载 {len(atmospheric_levels)} 个层边界 (Levels)")
#     print(f"✅ 成功计算 {len(atmospheric_profile)} 个层平均状态 (Layers)")

#     # 预计算HAPI参数
#     print("\n预计算HAPI物理参数...")
#     t_start = time.perf_counter()
#     all_layers_lines_params = Parallel(n_jobs=16)(
#         delayed(get_hapi_physical_params_new)(
#             MOLECULE,
#             GLOBAL_WN_MIN,
#             GLOBAL_WN_MAX,
#             layer["temperature_k"],
#             layer["pressure_pa"],
#             global_iso_ids=global_iso_ids,
#         )
#         for layer in tqdm(atmospheric_profile, desc="HAPI预计算")
#     )
#     print(f"✅ 预计算完成，耗时: {time.perf_counter() - t_start:.2f}秒")

#     # 加载模型
#     print("\n加载神经网络模型...")
#     statistics_pool = {
#         "lp": np.load(LP_STATS_PATH, allow_pickle=True).item(),
#         "hp": np.load(HP_STATS_PATH, allow_pickle=True).item(),
#     }
#     models = {
#         "lp": load_model(
#             LP_MODEL_PATH,
#             2,
#             [100, 500, 1000, 500],
#             TOTAL_POINTS_FULL,
#             DEVICE,
#             statistics_pool["lp"],
#         ),
#         "hp": load_model(
#             HP_MODEL_PATH,
#             2,
#             [100, 500, 1000, 500],
#             TOTAL_POINTS_FULL,
#             DEVICE,
#             statistics_pool["hp"],
#         ),
#     }
#     print("✅ 模型加载完成")

#     # GPU推理
#     print("\n开始GPU推理...")
#     t_infer_start = time.perf_counter()

#     # 分组
#     hp_layer_indices = [
#         i
#         for i, l in enumerate(atmospheric_profile)
#         if l["pressure_pa"] >= PRESSURE_THRESHOLD_PA
#     ]
#     lp_layer_indices = [
#         i
#         for i, l in enumerate(atmospheric_profile)
#         if l["pressure_pa"] < PRESSURE_THRESHOLD_PA
#     ]

#     all_layer_absorptions = [None] * len(atmospheric_profile)
#     mega_batch_size = 2

#     # 处理LP层
#     if lp_layer_indices:
#         print(f"处理低压层 ({len(lp_layer_indices)}层)...")
#         for batch_start in tqdm(range(0, len(lp_layer_indices), mega_batch_size)):
#             batch_end = min(batch_start + mega_batch_size, len(lp_layer_indices))
#             batch_indices = lp_layer_indices[batch_start:batch_end]

#             layer_gpu_results = process_mega_batch_gpu(
#                 batch_indices,
#                 all_layers_lines_params,
#                 models["lp"],
#                 base_grid_gpu,
#                 DEVICE,
#             )

#             batch_absorptions = Parallel(n_jobs=-1, backend="threading")(
#                 delayed(process_superposition_from_gpu)(
#                     p, w, global_wavenumber_grid, base_wavenumber_grid
#                 )
#                 for p, w in layer_gpu_results
#             )

#             for i, idx in enumerate(batch_indices):
#                 all_layer_absorptions[idx] = batch_absorptions[i]

#     # 处理HP层
#     if hp_layer_indices:
#         print(f"处理高压层 ({len(hp_layer_indices)}层)...")
#         for batch_start in tqdm(range(0, len(hp_layer_indices), mega_batch_size)):
#             batch_end = min(batch_start + mega_batch_size, len(hp_layer_indices))
#             batch_indices = hp_layer_indices[batch_start:batch_end]

#             layer_gpu_results = process_mega_batch_gpu(
#                 batch_indices,
#                 all_layers_lines_params,
#                 models["hp"],
#                 base_grid_gpu,
#                 DEVICE,
#             )

#             batch_absorptions = Parallel(n_jobs=-1, backend="threading")(
#                 delayed(process_superposition_from_gpu)(
#                     p, w, global_wavenumber_grid, base_wavenumber_grid
#                 )
#                 for p, w in layer_gpu_results
#             )

#             for i, idx in enumerate(batch_indices):
#                 all_layer_absorptions[idx] = batch_absorptions[i]

#     print(f"✅ GPU推理完成，耗时: {time.perf_counter() - t_infer_start:.2f}秒")

#     # 计算HAPI基准（带缓存，可跳过）
#     if SKIP_HAPI:
#         print("\n⚠️  跳过HAPI计算")
#         hapi_results = None
#     else:
#         import pickle
#         import hashlib

#         # 生成缓存文件名（基于配置参数）
#         cache_key = f"{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{PRESSURE_FILE}_{TEMPERATURE_FILE}"
#         cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
#         hapi_cache_path = f"cache/hapi_results_{cache_hash}.pkl"
#         os.makedirs("cache", exist_ok=True)

#         # 尝试加载缓存
#         if os.path.exists(hapi_cache_path):
#             print(f"\n发现HAPI缓存: {hapi_cache_path}")
#             print("正在加载...")
#             t_load_start = time.perf_counter()
#             with open(hapi_cache_path, "rb") as f:
#                 hapi_results = pickle.load(f)
#             print(
#                 f"✅ HAPI缓存加载完成，耗时: {time.perf_counter() - t_load_start:.2f}秒"
#             )
#         else:
#             print("\n未找到HAPI缓存，开始计算...")
#             t_hapi_start = time.perf_counter()
#             hapi_results = Parallel(n_jobs=8)(
#                 delayed(calculate_hapi_benchmark_new)(
#                     MOLECULE,
#                     global_wavenumber_grid,
#                     layer["temperature_k"],
#                     layer["pressure_pa"],
#                     GLOBAL_WN_MIN,
#                     GLOBAL_WN_MAX,
#                 )
#                 for layer in tqdm(atmospheric_profile, desc="HAPI计算")
#             )
#             print(f"✅ HAPI完成，耗时: {time.perf_counter() - t_hapi_start:.2f}秒")

#             # 保存缓存
#             print(f"正在保存HAPI缓存...")
#             with open(hapi_cache_path, "wb") as f:
#                 pickle.dump(hapi_results, f, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f"✅ HAPI缓存已保存: {hapi_cache_path}")

#     # 保存数据
#     save_to_hdf5(
#         OUTPUT_H5,
#         atmospheric_profile,
#         all_layer_absorptions,
#         hapi_results,
#         global_wavenumber_grid,
#         MOLECULE,
#     )

#     print("=" * 80)
#     print("全部完成！")
#     print("=" * 80)
