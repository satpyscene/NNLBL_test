"""
单层吸收系数计算核心库
包含：模型结构、工具函数、HAPI参数获取、物理恢复、叠加、基准计算
"""
import time
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import numba
from numba import get_num_threads, get_thread_id
from tqdm import tqdm

from hapi import (
    db_begin,
    fetch,
    absorptionCoefficient_Voigt,
    calculateProfileParametersVoigt,
    LOCAL_TABLE_CACHE,
    PYTIPS,
    tableList,
)


# --- 模型结构定义 ---
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc5 = nn.Linear(hidden_size[3], output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out


# --- 工具函数 ---
def create_non_uniform_grid(nu0, wing_cm, total_points, concentration_factor):
    """创建以nu0为中心的非均匀波数网格"""
    x_uniform = np.linspace(-1, 1, total_points)
    x_non_uniform = np.sinh(x_uniform * concentration_factor) / np.sinh(
        concentration_factor
    )
    wavenumber_grid = nu0 + wing_cm * x_non_uniform
    return wavenumber_grid


# --- HAPI参数获取函数 ---
def get_hapi_physical_params(
    molecule_name, wavenumber_min, wavenumber_max, temperature_k, pressure_pa
):
    """使用HAPI为给定P, T条件动态计算所有相关谱线的物理参数"""
    MOLECULE_IDS = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
    molecule_id = MOLECULE_IDS[molecule_name]
    pressure_atm = pressure_pa / 101325.0

    db_path = f"data/{molecule_name}_hapi"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    db_begin(db_path)

    table_name = f"{molecule_name}_{wavenumber_min}_{wavenumber_max}"

    if table_name not in tableList():
        fetch(table_name, molecule_id, 1, wavenumber_min, wavenumber_max)

    DATA_DICT = LOCAL_TABLE_CACHE[table_name]["data"]
    num_total_lines = len(DATA_DICT["nu"])

    all_lines_params = []
    for i in range(num_total_lines):
        trans = {param: DATA_DICT[param][i] for param in DATA_DICT}
        trans["T"], trans["p"] = temperature_k, pressure_atm
        trans["T_ref"], trans["p_ref"] = 296.0, 1.0
        trans["Diluent"] = {"air": 1.0}
        trans["SigmaT"] = PYTIPS(trans["molec_id"], trans["local_iso_id"], trans["T"])
        trans["SigmaT_ref"] = PYTIPS(
            trans["molec_id"], trans["local_iso_id"], trans["T_ref"]
        )
        line_calc_params = calculateProfileParametersVoigt(TRANS=trans)
        all_lines_params.append(
            {
                "gamma_d": line_calc_params["GammaD"],
                "gamma_l": line_calc_params["Gamma0"],
                "delta_0": line_calc_params["Delta0"],
                "nu0": trans["nu"],
                "S": line_calc_params["Sw"],
            }
        )

    return all_lines_params


# --- 物理恢复函数 ---
def physical_restore_numpy(
    predicted_base_profiles_normalized,
    y_mean,
    y_std,
    s_values,
    nu0_values,
    delta_0_values,
    base_wavenumber_grid,
):
    """使用纯 NumPy 向量化操作进行物理恢复"""
    s_values = np.asarray(s_values)
    nu0_values = np.asarray(nu0_values)
    delta_0_values = np.asarray(delta_0_values)

    y_log_scale = predicted_base_profiles_normalized * y_std + y_mean
    y_physical_scale = 10**y_log_scale
    final_profiles = y_physical_scale * s_values[:, np.newaxis]

    total_shifts = (nu0_values + delta_0_values) - 1000.0
    final_wavenumber_grids = base_wavenumber_grid + total_shifts[:, np.newaxis]

    return final_profiles, final_wavenumber_grids


# --- Numba加速叠加函数 ---
@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def perform_superposition_cpu_safe(
    final_profiles, final_wavenumber_grids, global_wavenumber_grid
):
    """在 CPU 上进行并行叠加的安全、高效实现"""
    num_lines = final_profiles.shape[0]
    num_global_points = global_wavenumber_grid.shape[0]
    num_threads = get_num_threads()
    private_absorptions = np.zeros((num_threads, num_global_points), dtype=np.float64)

    for i in numba.prange(num_lines):
        thread_id = get_thread_id()
        local_wn = final_wavenumber_grids[i]
        y_pred = final_profiles[i]

        start_idx = np.searchsorted(global_wavenumber_grid, local_wn[0], side="left")
        end_idx = np.searchsorted(global_wavenumber_grid, local_wn[-1], side="right")

        if start_idx >= end_idx:
            continue

        for j in range(start_idx, end_idx):
            global_wn_point = global_wavenumber_grid[j]
            local_idx = np.searchsorted(local_wn, global_wn_point)

            if local_idx == 0 or local_idx >= len(local_wn):
                continue

            x1, x2 = local_wn[local_idx - 1], local_wn[local_idx]
            y1, y2 = y_pred[local_idx - 1], y_pred[local_idx]

            if x2 == x1:
                interpolated_value = y1
            else:
                interpolated_value = y1 + (global_wn_point - x1) * (y2 - y1) / (x2 - x1)

            if interpolated_value > 0:
                private_absorptions[thread_id, j] += interpolated_value

    total_absorption = np.sum(private_absorptions, axis=0)
    return total_absorption


# --- HAPI基准计算 ---
def calculate_hapi_benchmark(
    molecule_name,
    wavenumber_grid,
    temperature_k,
    pressure_pa,
    wavenumber_min,
    wavenumber_max,
):
    """计算HAPI基准吸收系数"""
    hapi_start_time = time.perf_counter()
    pressure_atm = pressure_pa / 101325.0
    table_name = f"{molecule_name}_{wavenumber_min}_{wavenumber_max}"

    nu_hapi, coef_hapi = absorptionCoefficient_Voigt(
        SourceTables=table_name,
        WavenumberGrid=wavenumber_grid,
        Environment={"T": temperature_k, "p": pressure_atm},
        WavenumberWing=25,
    )
    hapi_duration = time.perf_counter() - hapi_start_time
    return nu_hapi, coef_hapi, hapi_duration


# --- 模型加载 ---
def load_model(model_path, input_size, hidden_size, output_size, device):
    """加载训练好的模型"""
    model = SimpleNet(input_size, hidden_size, output_size)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# --- 核心计算函数 ---
def compute_absorption_for_condition(
    molecule_name,
    wavenumber_min,
    wavenumber_max,
    wavenumber_step,
    pressure_pa,
    temperature_k,
    model,
    statistics,
    device,
    wing_size=25.0,
    total_points=5001,
    concentration_factor=6.0,
):
    """为单个条件计算模型预测和HAPI基准的吸收光谱"""

    # 1. 获取HAPI物理参数
    all_lines_params = get_hapi_physical_params(
        molecule_name, wavenumber_min, wavenumber_max, temperature_k, pressure_pa
    )
    if not all_lines_params:
        return None

    # 2. 模型推理
    model_start = time.perf_counter()

    model_inputs_raw = np.array(
        [[line["gamma_d"], line["gamma_l"]] for line in all_lines_params]
    )
    model_inputs_normalized = (model_inputs_raw - statistics["x_mean"]) / statistics[
        "x_std"
    ]

    input_tensor = torch.tensor(model_inputs_normalized, dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_base_profiles_normalized = model(input_tensor).cpu().numpy()

    s_values = np.array([line["S"] for line in all_lines_params])
    nu0_values = np.array([line["nu0"] for line in all_lines_params])
    delta_0_values = np.array([line["delta_0"] for line in all_lines_params])

    base_wavenumber_grid = create_non_uniform_grid(
        1000.0, wing_size, total_points, concentration_factor
    )

    final_profiles, final_wavenumber_grids = physical_restore_numpy(
        predicted_base_profiles_normalized,
        statistics["y_mean"],
        statistics["y_std"],
        s_values,
        nu0_values,
        delta_0_values,
        base_wavenumber_grid,
    )

    global_wavenumber_grid = np.arange(wavenumber_min, wavenumber_max, wavenumber_step)
    total_absorption_model = perform_superposition_cpu_safe(
        final_profiles, final_wavenumber_grids, global_wavenumber_grid
    )

    model_time = time.perf_counter() - model_start

    # 3. HAPI基准计算
    hapi_wn, hapi_coef, hapi_time = calculate_hapi_benchmark(
        molecule_name,
        global_wavenumber_grid,
        temperature_k,
        pressure_pa,
        wavenumber_min,
        wavenumber_max,
    )

    return {
        "wavenumber": global_wavenumber_grid,
        "model_absorption": total_absorption_model,
        "hapi_absorption": hapi_coef,
        "model_time": model_time,
        "hapi_time": hapi_time,
        "molecule": molecule_name,
        "pressure": pressure_pa,
        "temperature": temperature_k,
        "wn_min": wavenumber_min,
        "wn_max": wavenumber_max,
    }
