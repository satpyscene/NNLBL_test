import sys
import os
import time
import numpy as np
import torch
import pickle
import hashlib
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from .run_inference_and_save import (
    create_non_uniform_grid,
    get_hapi_physical_params_new,
    load_model,
    process_mega_batch_gpu,
    process_superposition_from_gpu,
    calculate_hapi_benchmark_new,
    save_to_hdf5,
)
from .mt_ckd_h2o import MTCKD_H2O


def validate_single_molecule_iso_list(global_iso_ids):
    """
    检查 target_iso_list 是否只包含同一种分子的同位素编号
    """
    TOP7_ISO_TABLE = {
        "H2O": [1, 2, 3, 4, 5, 6, 129],
        "CO2": [7, 8, 9, 10, 11, 12, 13, 14],
        "O3": [16, 17, 18, 19, 20],
        "N2O": [21, 22, 23, 24, 25],
        "CO": [26, 27, 28, 29, 30, 31],
        "CH4": [32, 33, 34, 35],
        "O2": [36, 37, 38],
    }

    if not global_iso_ids:
        raise ValueError("❌ target_iso_list 不能为空")

    found_molecules = set()

    for gid in global_iso_ids:
        matched = False
        for mol, iso_list in TOP7_ISO_TABLE.items():
            if gid in iso_list:
                found_molecules.add(mol)
                matched = True
                break
        if not matched:
            raise ValueError(f"❌ 同位素编号 {gid} 不在支持的 TOP7 分子表中")

    if len(found_molecules) > 1:
        raise ValueError(
            f"❌ 不允许混合不同分子的同位素编号: 检测到 {sorted(found_molecules)}"
        )

    # 返回分子名，供需要时使用
    return next(iter(found_molecules))


def generate_molecule_label(global_iso_ids):
    # ==========================================
    # 1. 你的专属数据表
    # ==========================================
    TOP7_ISO_TABLE = {
        "H2O": [1, 2, 3, 4, 5, 6, 129],
        "CO2": [7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122],
        "O3": [16, 17, 18, 19, 20],
        "N2O": [21, 22, 23, 24, 25],
        "CO": [26, 27, 28, 29, 30, 31],
        "CH4": [32, 33, 34, 35],
        "O2": [36, 37, 38],
    }

    if not global_iso_ids:
        return "Unknown_Empty"

    # 用于临时存储找到的分子及其包含的局部索引
    # 结构: { "CO2": [1, 2], "H2O": [1] } (这里存的是列表中的下标+1)
    found_map = {}

    # ==========================================
    # 2. 核心查找逻辑
    # ==========================================
    for gid in global_iso_ids:
        found = False
        for mol_name, iso_list in TOP7_ISO_TABLE.items():
            if gid in iso_list:
                if mol_name not in found_map:
                    found_map[mol_name] = []

                # 获取该ID在列表中的位置 (从1开始计数)，作为"IsoX"的标号
                local_index = iso_list.index(gid) + 1
                found_map[mol_name].append(local_index)
                found = True
                break

        if not found:
            print(f"⚠️ 警告: 全局编号 {gid} 不在 TOP7 表格中，将被忽略。")

    # ==========================================
    # 3. 生成描述性字符串
    # ==========================================
    name_segments = []

    # 按表格定义的顺序来生成文件名，保持整洁
    for mol_name in TOP7_ISO_TABLE.keys():
        if mol_name in found_map:
            indices = sorted(found_map[mol_name])
            total_count_in_table = len(TOP7_ISO_TABLE[mol_name])

            # --- 命名规则 ---
            if len(indices) == total_count_in_table:
                # 规则1: 如果包含该分子全部同位素 -> "_All"
                label = f"{mol_name}_All"
            elif len(indices) == 1 and indices[0] == 1:
                # 规则2: 如果只有第1个(通常是主同位素) -> "_Major"
                label = f"{mol_name}_Major"
            else:
                # 规则3: 其他情况 -> "_Iso1-2-5"
                str_indices = "-".join(map(str, indices))
                label = f"{mol_name}_Iso{str_indices}"

            name_segments.append(label)

    if not name_segments:
        return "Unknown_NoMatch"

    return "_".join(name_segments)


# ------------------------------------------------------------------------------
# 输入配置校验函数
# ------------------------------------------------------------------------------
def validate_user_config(
    run_mode,
    single_params,
    profile_params,
    spectral_config,
    target_iso_list,
):
    """
    对用户输入配置进行前置校验（fail-fast）
    只检查“结构与显式声明”，不做任何单位转换或数值物理判断
    """

    # ---------- Run mode ----------
    if run_mode not in ("SINGLE", "PROFILE"):
        raise ValueError(
            f"❌ RUN_MODE 必须是 'SINGLE' 或 'PROFILE'，当前为: {run_mode}"
        )

    # ---------- Target isotopes ----------
    if not target_iso_list:
        raise ValueError("❌ TARGET_ISO_LIST 不能为空")

    if not all(isinstance(i, int) for i in target_iso_list):
        raise ValueError("❌ TARGET_ISO_LIST 中必须全部为整数（HITRAN isotope ID）")

    # ---------- Spectral config ----------
    for key in ("min", "max", "step"):
        if key not in spectral_config:
            raise ValueError(f"❌ SPECTRAL_CONFIG 缺少字段: '{key}'")

    if spectral_config["min"] >= spectral_config["max"]:
        raise ValueError("❌ SPECTRAL_CONFIG 中 min 必须小于 max")

    # ---------- SINGLE mode ----------
    if run_mode == "SINGLE":
        if not single_params:
            raise ValueError("❌ RUN_MODE=SINGLE 时，必须提供 SINGLE_PARAMS")

        required_single_keys = (
            "p_pa",
            "p_hpa",
            "t_k",
            "t_c",
            "vmr",
            "vmr_ppmv",
        )

        if not any(k in single_params for k in ("p_pa", "p_hpa")):
            raise ValueError("❌ SINGLE_PARAMS 必须包含 p_pa 或 p_hpa")

        if not any(k in single_params for k in ("t_k", "t_c")):
            raise ValueError("❌ SINGLE_PARAMS 必须包含 t_k 或 t_c")

        if not any(k in single_params for k in ("vmr", "vmr_ppmv")):
            raise ValueError("❌ SINGLE_PARAMS 必须包含 vmr 或 vmr_ppmv")

    # ---------- PROFILE mode ----------
    if run_mode == "PROFILE":
        if not profile_params:
            raise ValueError("❌ RUN_MODE=PROFILE 时，必须提供 PROFILE_PARAMS")

        required_profile_keys = ("p_file", "p_unit", "t_file", "t_unit")
        for k in required_profile_keys:
            if k not in profile_params:
                raise ValueError(f"❌ PROFILE_PARAMS 缺少必要字段: '{k}'")

        if "vmr_file" in profile_params and "vmr_unit" not in profile_params:
            raise ValueError("❌ 提供 vmr_file 时，必须同时提供 vmr_unit")


def _load_and_standardize_data(mode, single_cfg, profile_cfg, base_dir):
    """
    数据加载 + 单位校验 + 标准化
    要求用户在配置中显式给出单位，否则直接报错
    内部统一转换为:
        Pressure : Pa
        Temperature : K
        VMR : volume mixing ratio (无量纲)
    """
    if mode == "SINGLE":
        # ---------- Pressure ----------
        if "p_pa" in single_cfg:
            p = np.array([single_cfg["p_pa"]], dtype=float)
        elif "p_hpa" in single_cfg:
            p = np.array([single_cfg["p_hpa"] * 100.0], dtype=float)
        else:
            raise ValueError("❌ SINGLE 模式必须提供气压单位: p_pa 或 p_hpa")

        # ---------- Temperature ----------
        if "t_k" in single_cfg:
            t = np.array([single_cfg["t_k"]], dtype=float)
        elif "t_c" in single_cfg:
            t = np.array([single_cfg["t_c"] + 273.15], dtype=float)
        else:
            raise ValueError("❌ SINGLE 模式必须提供温度单位: t_k 或 t_c")

        # ---------- VMR ----------
        if "vmr" in single_cfg:
            v = np.array([single_cfg["vmr"]], dtype=float)
        elif "vmr_ppmv" in single_cfg:
            v = np.array([single_cfg["vmr_ppmv"] * 1e-6], dtype=float)
        else:
            raise ValueError("❌ SINGLE 模式必须提供 vmr 或 vmr_ppmv")

        suffix = f"{int(p[0])}_{int(t[0])}"

    elif mode == "PROFILE":
        prof_dir = base_dir / profile_cfg.get("dir", "")

        # ---------- Pressure ----------
        p_path = prof_dir / profile_cfg["p_file"]
        if not p_path.exists():
            raise FileNotFoundError(f"❌ 找不到气压文件: {p_path}")

        p_unit = profile_cfg.get("p_unit", None)
        if p_unit is None:
            raise ValueError("❌ PROFILE 模式必须声明气压单位 p_unit")

        p_raw = np.loadtxt(p_path)
        if p_unit == "Pa":
            p = p_raw
        elif p_unit == "hPa":
            p = p_raw * 100.0
        else:
            raise ValueError(f"❌ 不支持的气压单位: {p_unit}")

        # ---------- Temperature ----------
        t_path = prof_dir / profile_cfg["t_file"]
        if not t_path.exists():
            raise FileNotFoundError(f"❌ 找不到温度文件: {t_path}")

        t_unit = profile_cfg.get("t_unit", None)
        if t_unit is None:
            raise ValueError("❌ PROFILE 模式必须声明温度单位 t_unit")

        t_raw = np.loadtxt(t_path)
        if t_unit == "K":
            t = t_raw
        elif t_unit == "C":
            t = t_raw + 273.15
        else:
            raise ValueError(f"❌ 不支持的温度单位: {t_unit}")

        # ---------- VMR ----------
        vmr_path = prof_dir / profile_cfg.get("vmr_file", "")
        vmr_unit = profile_cfg.get("vmr_unit", "vmr")  # 默认 vmr

        if vmr_path.exists():
            v_raw = np.loadtxt(vmr_path)
            if vmr_unit == "vmr":
                v = v_raw
            elif vmr_unit == "ppmv":
                v = v_raw * 1e-6
            else:
                raise ValueError(f"❌ 不支持的 VMR 单位: {vmr_unit}")
        else:
            v = np.zeros_like(p)

        suffix = profile_cfg.get("name_tag", "PROFILE")
    else:
        raise ValueError(f"未知模式: {mode}")

    return p, t, v, suffix


# ==============================================================================
# 1. 全局配置常量 (System Configuration)
#    这些参数定义了环境、模型路径和物理常数，通常在main内部直接调用
# ==============================================================================

# --- 路径配置 (服务器/环境相关) ---


def NNLBL_API(
    target_iso_list,
    spectral_config,
    input_mode,
    single_config,
    profile_config,
    path_config,
    enable_continuum=True,
    skip_hapi=False,
    gamma_l_threshold=None,  # 新增：洛伦兹半高半宽阈值（cm⁻¹）
):
    """
    NNLBL 的高级封装接口。负责数据加载、路径推导，最后调用核心算法。
    """
    base_dir = Path(path_config["base_dir"])

    # --- A. 自动加载数据 ---
    # (把之前的 load_input_data 逻辑搬到这里)
    p_vals, t_vals, vmr_vals, file_suffix = _load_and_standardize_data(
        input_mode, single_config, profile_config, base_dir
    )

    # --- B. 自动推导模型路径 ---
    # (假设模型都在规定的文件夹下，不需要用户一个个传)
    model_dir = base_dir / path_config["model_dir"]
    model_paths = {
        "hp_m": str(model_dir / "voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth"),
        "hp_s": str(model_dir / "voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy"),
        "lp_m": str(
            model_dir / "voigt_model_best_lp_Full-nonuniform-n0_1000_noshift.pth"
        ),
        "lp_s": str(model_dir / "lp_voigt_stats_Full-nonuniform-n0_1000_noshift.npy"),
    }

    # --- C0. 校验同位素列表合法性（禁止跨分子混合） ---
    validated_molecule = validate_single_molecule_iso_list(target_iso_list)
    print(f"同位素校验通过，目标分子: {validated_molecule}")

    # --- C. 自动生成输出文件名 ---
    mol_label = generate_molecule_label(target_iso_list)
    output_dir = base_dir / path_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{mol_label}_{spectral_config['min']}_{spectral_config['max']}_{spectral_config['step']}_{file_suffix}.h5"
    output_path = str(output_dir / filename)

    print(f"任务标识: {mol_label}")
    print(f"数据加载: {len(p_vals)} 层")
    print(f"输出目标: {output_path}")

    # --- D. 检查 MT-CKD ---
    mtckd_path = str(base_dir / path_config["mtckd_file"])
    if 1 in target_iso_list and not os.path.exists(mtckd_path):
        print(f"⚠️ 警告: 计算水汽但找不到 MT-CKD 文件: {mtckd_path}")

    # --- E. 调用内核 (原来的 NNLBL_main) ---
    NNLBL_main(
        MOLECULE=mol_label,
        GLOBAL_WN_MIN=spectral_config["min"],
        GLOBAL_WN_MAX=spectral_config["max"],
        GLOBAL_WN_STEP=spectral_config["step"],
        input_pressures=p_vals,
        input_temperatures=t_vals,
        input_vmrs=vmr_vals,
        mtckd_path=mtckd_path,
        output_path=output_path,
        HP_MODEL_PATH=model_paths["hp_m"],
        HP_STATS_PATH=model_paths["hp_s"],
        LP_MODEL_PATH=model_paths["lp_m"],
        LP_STATS_PATH=model_paths["lp_s"],
        skip_hapi=skip_hapi,
        global_iso_ids=target_iso_list,
        enable_continuum=enable_continuum,
        gamma_l_threshold=gamma_l_threshold,  # 新增：传递gamma_l阈值
    )


# ==============================================================================
# 2. 主逻辑函数 (Core Logic)
#    这里只定义逻辑，不包含具体的数据文件路径。
#    输入：气压(Pa), 温度(K), 输出路径, 标志位
# ==============================================================================
def NNLBL_main(
    MOLECULE,
    GLOBAL_WN_MIN,
    GLOBAL_WN_MAX,
    GLOBAL_WN_STEP,
    input_pressures,
    input_temperatures,
    input_vmrs,
    mtckd_path,
    output_path,
    HP_MODEL_PATH="NNmodel&stats/voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth",
    HP_STATS_PATH="NNmodel&stats/voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy",
    LP_MODEL_PATH="NNmodel&stats/voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth",
    LP_STATS_PATH="NNmodel&stats/voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy",
    skip_hapi=False,
    global_iso_ids=None,
    enable_continuum=True,
    gamma_l_threshold=None,  # 新增：洛伦兹半高半宽阈值（用于替代气压阈值）
):
    print("!" * 80)
    print(
        "致谢，NNLBL算法的线参数订正模块，训练数据，以及结果验证均来自HAPI！HAPI version: 1.2.2.4"
    )
    print("!" * 80)

    # 模型分界参数
    # 新版：优先使用 gamma_l_threshold（洛伦兹半高半宽阈值）
    # 如果未指定 gamma_l_threshold，则回退到旧的气压阈值模式
    PRESSURE_THRESHOLD_PA = (
        2200.0  # 气压分界线（旧版，仅在 gamma_l_threshold=None 时使用）
    )

    WING_SIZE = 25.0
    TOTAL_POINTS_FULL = 5001
    CONCENTRATION_FACTOR = 6.0

    # 判断使用哪种划分模式
    if gamma_l_threshold is not None:
        print(f">> 模型划分模式: 洛伦兹半高半宽阈值 = {gamma_l_threshold:.6f} cm⁻¹")
        print("   - γ_L ≥ 阈值 → HP模型（高压/宽谱线）")
        print("   - γ_L < 阈值 → LP模型（低压/窄谱线）")
        use_gamma_l_mode = True
    else:
        print(f">> 模型划分模式: 气压阈值 = {PRESSURE_THRESHOLD_PA} Pa（传统模式）")
        print("   - P ≥ 阈值 → HP模型")
        print("   - P < 阈值 → LP模型")
        use_gamma_l_mode = False

    print("=" * 80)
    print(f"启动推理流程 | 输出目标: {output_path}")
    print("=" * 80)

    # ---------------------------------------------------
    # A. 环境与网格初始化
    # ---------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 全局均匀网格 (用于最终输出)
    global_wavenumber_grid = np.arange(
        GLOBAL_WN_MIN, GLOBAL_WN_MAX + GLOBAL_WN_STEP * 0.5, GLOBAL_WN_STEP
    )

    # 基础非均匀网格 (用于神经网络输入)
    base_wavenumber_grid = create_non_uniform_grid(
        1000.0, WING_SIZE, TOTAL_POINTS_FULL, CONCENTRATION_FACTOR
    )
    base_grid_gpu = torch.tensor(
        base_wavenumber_grid, dtype=torch.float64, device=DEVICE
    )

    # ---------------------------------------------------
    # B. 输入数据标准化 (处理单层/多层逻辑)
    # ---------------------------------------------------

    # 确保输入是 numpy 数组
    p_data = np.array(input_pressures, dtype=float)
    t_data = np.array(input_temperatures, dtype=float)
    if input_vmrs is None:
        # 如果是单层，给 0.0；如果是多层，给全 0
        # 这里简单处理，假设外层已经处理好了 input_vmrs 不为 None
        vmr_data = np.zeros_like(p_data, dtype=float)
    else:
        vmr_data = np.array(input_vmrs, dtype=float)
    # 自动判断：如果是单个数字(0维)，转为1维数组；如果是数组，保持原样
    if p_data.ndim == 0:
        p_data = p_data.reshape(1)
        t_data = t_data.reshape(1)
        vmr_data = vmr_data.reshape(1)  # [修改点 2] VMR 也要 reshape，保证可迭代
        print(">> 模式检测: 单层输入 (Single Layer)")
    else:
        print(f">> 模式检测: 大气廓线输入 (Profile, {len(p_data)} Layers)")

    # 校验维度一致性
    if p_data.shape != t_data.shape:
        raise ValueError("❌ 错误: 气压和温度数据的长度不一致！")

    # 必须确保 P, T, VMR 三者长度完全一样，否则 zip 会丢失数据
    if not (p_data.shape == t_data.shape == vmr_data.shape):
        raise ValueError(
            f"❌ 错误: 输入数据长度不一致！\n"
            f"P: {p_data.shape}, T: {t_data.shape}, VMR: {vmr_data.shape}"
        )
    # 直接使用输入值作为层的状态
    atmospheric_profile = [
        {"layer": i, "pressure_pa": p, "temperature_k": t, "vmr": vmr}
        for i, (p, t, vmr) in enumerate(zip(p_data, t_data, vmr_data))
    ]
    print(f"大气分子参数加载: 共 {len(atmospheric_profile)} 层")

    # ---------------------------------------------------
    # C. 预计算 HAPI 参数
    # ---------------------------------------------------
    print("\n根据实际温压订正Voigt线型参数...")
    t_start = time.perf_counter()

    all_layers_lines_params = Parallel(n_jobs=16)(
        delayed(get_hapi_physical_params_new)(
            MOLECULE,
            GLOBAL_WN_MIN,
            GLOBAL_WN_MAX,
            layer["temperature_k"],
            layer["pressure_pa"],
            global_iso_ids=global_iso_ids,
            vmr=layer["vmr"],
        )
        for layer in tqdm(atmospheric_profile, desc="吸收线参数计算进程")
    )
    print(f"吸收线参数全部计算完成，耗时: {time.perf_counter() - t_start:.2f}秒")

    # ---------------------------------------------------
    # D. 加载神经网络模型
    # ---------------------------------------------------
    # print("\n加载神经网络模型...")
    statistics_pool = {
        "lp": np.load(LP_STATS_PATH, allow_pickle=True).item(),
        "hp": np.load(HP_STATS_PATH, allow_pickle=True).item(),
    }
    models = {
        "lp": load_model(
            LP_MODEL_PATH,
            2,
            [100, 500, 1000, 500],
            TOTAL_POINTS_FULL,
            DEVICE,
            statistics_pool["lp"],
        ),
        "hp": load_model(
            HP_MODEL_PATH,
            2,
            [100, 500, 1000, 500],
            TOTAL_POINTS_FULL,
            DEVICE,
            statistics_pool["hp"],
        ),
    }
    print("两个神经网络模型已加载至目标设备")

    # ---------------------------------------------------
    # E. GPU 推理
    # ---------------------------------------------------
    t_infer_start = time.perf_counter()
    all_layer_absorptions = [None] * len(atmospheric_profile)
    mega_batch_size = 2

    if use_gamma_l_mode:
        # ========================================
        # 新模式：基于洛伦兹半高半宽 (gamma_l) 划分
        # ========================================
        print("\n开始GPU推理（基于γ_L阈值的混合模型模式）...")
        print(f"每层将根据谱线的γ_L分别使用HP和LP模型")

        all_layer_indices = list(range(len(atmospheric_profile)))

        # 遍历所有层，每层分别处理HP和LP谱线
        for batch_start in tqdm(
            range(0, len(all_layer_indices), mega_batch_size), desc="处理各层"
        ):
            batch_end = min(batch_start + mega_batch_size, len(all_layer_indices))
            batch_indices = all_layer_indices[batch_start:batch_end]

            # 对当前batch中的每一层，分别处理HP和LP部分
            for idx in batch_indices:
                # --- 处理该层的HP谱线（gamma_l >= threshold）---
                layer_gpu_results_hp = process_mega_batch_gpu(
                    [idx],  # 单层
                    all_layers_lines_params,
                    models["hp"],
                    base_grid_gpu,
                    DEVICE,
                    gamma_l_threshold=gamma_l_threshold,
                    use_high_gamma=True,  # 使用HP模型处理高gamma_l谱线
                )

                # --- 处理该层的LP谱线（gamma_l < threshold）---
                layer_gpu_results_lp = process_mega_batch_gpu(
                    [idx],  # 单层
                    all_layers_lines_params,
                    models["lp"],
                    base_grid_gpu,
                    DEVICE,
                    gamma_l_threshold=gamma_l_threshold,
                    use_high_gamma=False,  # 使用LP模型处理低gamma_l谱线
                )

                # --- 插值并合并HP和LP的结果 ---
                # HP部分
                profiles_hp, wn_grids_hp = layer_gpu_results_hp[0]
                if profiles_hp is not None:
                    absorption_hp = process_superposition_from_gpu(
                        profiles_hp,
                        wn_grids_hp,
                        global_wavenumber_grid,
                        base_wavenumber_grid,
                    )
                else:
                    absorption_hp = np.zeros_like(global_wavenumber_grid)

                # LP部分
                profiles_lp, wn_grids_lp = layer_gpu_results_lp[0]
                if profiles_lp is not None:
                    absorption_lp = process_superposition_from_gpu(
                        profiles_lp,
                        wn_grids_lp,
                        global_wavenumber_grid,
                        base_wavenumber_grid,
                    )
                else:
                    absorption_lp = np.zeros_like(global_wavenumber_grid)

                # 合并（简单相加，因为谱线不重叠）
                all_layer_absorptions[idx] = absorption_hp + absorption_lp

    else:
        # ========================================
        # 旧模式：基于气压阈值划分（向后兼容）
        # ========================================
        print("\n开始GPU推理（传统气压阈值模式）...")

        # 根据气压阈值分组
        hp_layer_indices = [
            i
            for i, l in enumerate(atmospheric_profile)
            if l["pressure_pa"] >= PRESSURE_THRESHOLD_PA
        ]
        lp_layer_indices = [
            i
            for i, l in enumerate(atmospheric_profile)
            if l["pressure_pa"] < PRESSURE_THRESHOLD_PA
        ]

        # --- 处理低压层 (LP) ---
        if lp_layer_indices:
            print(f"处理低压层 ({len(lp_layer_indices)}层)...")
            for batch_start in tqdm(range(0, len(lp_layer_indices), mega_batch_size)):
                batch_end = min(batch_start + mega_batch_size, len(lp_layer_indices))
                batch_indices = lp_layer_indices[batch_start:batch_end]

                layer_gpu_results = process_mega_batch_gpu(
                    batch_indices,
                    all_layers_lines_params,
                    models["lp"],
                    base_grid_gpu,
                    DEVICE,
                    gamma_l_threshold=None,  # 旧模式不使用gamma_l过滤
                    use_high_gamma=True,
                )

                # 后处理：插值回全局网格
                batch_absorptions = Parallel(n_jobs=-1, backend="threading")(
                    delayed(process_superposition_from_gpu)(
                        p, w, global_wavenumber_grid, base_wavenumber_grid
                    )
                    for p, w in layer_gpu_results
                )

                for i, idx in enumerate(batch_indices):
                    all_layer_absorptions[idx] = batch_absorptions[i]

        # --- 处理高压层 (HP) ---
        if hp_layer_indices:
            print(f"处理高压层 ({len(hp_layer_indices)}层)...")
            for batch_start in tqdm(range(0, len(hp_layer_indices), mega_batch_size)):
                batch_end = min(batch_start + mega_batch_size, len(hp_layer_indices))
                batch_indices = hp_layer_indices[batch_start:batch_end]

                layer_gpu_results = process_mega_batch_gpu(
                    batch_indices,
                    all_layers_lines_params,
                    models["hp"],
                    base_grid_gpu,
                    DEVICE,
                    gamma_l_threshold=None,  # 旧模式不使用gamma_l过滤
                    use_high_gamma=True,
                )

                # 后处理：插值回全局网格
                batch_absorptions = Parallel(n_jobs=-1, backend="threading")(
                    delayed(process_superposition_from_gpu)(
                        p, w, global_wavenumber_grid, base_wavenumber_grid
                    )
                    for p, w in layer_gpu_results
                )

                for i, idx in enumerate(batch_indices):
                    all_layer_absorptions[idx] = batch_absorptions[i]

    print(
        f"NNLBL吸收截面光谱计算结束，耗时: {time.perf_counter() - t_infer_start:.2f}秒"
    )

    continuum_cache = []

    # 1. 判断是否需要计算
    is_h2o_task = "H2O" in MOLECULE or (global_iso_ids and 1 in global_iso_ids)
    should_run_continuum = enable_continuum and is_h2o_task

    if should_run_continuum:
        print("\n" + "=" * 60)
        print("启动 MT-CKD 水汽连续吸收计算模块")
        if not os.path.exists(mtckd_path):
            print(f"❌ 错误: 找不到 MT-CKD 数据文件: {mtckd_path}")
            print(">> 跳过连续吸收叠加，结果将仅包含线吸收！")
            should_run_continuum = False  # 强制关闭
        else:
            t_cont_start = time.perf_counter()
            # 2. 实例化模型 (只加载一次文件，提高效率)
            print(f"加载 MT-CKD 模型: {os.path.basename(mtckd_path)}")
            mtckd_model = MTCKD_H2O(mtckd_path)

            print(f"正在计算 {len(atmospheric_profile)} 层的连续吸收...")

            # 3. 逐层计算
            for i, layer in enumerate(atmospheric_profile):
                p_pa = layer["pressure_pa"]
                t_k = layer["temperature_k"]
                vmr = layer["vmr"]

                # 单位转换: Pa -> hPa
                p_hpa = p_pa / 100.0

                # 如果 VMR 极小，连续吸收可忽略 (避免计算开销)
                if vmr < 1e-9:

                    cont_total = np.zeros_like(global_wavenumber_grid)
                    # print("vmr=", vmr)
                    # import matplotlib.pyplot as plt

                    # plt.plot(cont_total)
                else:
                    # [核心调用] 使用你提供的接口
                    # 注意: global_wavenumber_grid 必须与 returned nu 一致
                    # mtckd 模块内部通常会重新生成网格，这里我们只取吸收值

                    _, val_self, val_for = mtckd_model.get_absorption(
                        p_hpa,  # hPa
                        t_k,  # K
                        vmr,  # VMR
                        GLOBAL_WN_MIN,
                        GLOBAL_WN_MAX,
                        GLOBAL_WN_STEP,
                        radflag=True,
                    )
                    # 叠加 Self 和 Foreign 分量
                    cont_total = val_self + val_for

                    # [单位校验警示]
                    # 假设 mtckd 返回的是与 NN 输出一致的单位 (通常是截面 cm2 或 系数 cm-1)
                    # 如果 NN 输出是截面 (cm2/molecule)，确保 cont_total 也是截面！

                # 4. 暂存结果 (给后续 HAPI Benchmark 用)
                continuum_cache.append(cont_total)

                # 5. 叠加到 NN 预测结果上 (In-place addition)
                if all_layer_absorptions[i] is not None:

                    all_layer_absorptions[i] = all_layer_absorptions[i] + cont_total

            print(f"连续吸收处理完成，耗时: {time.perf_counter() - t_cont_start:.2f}秒")
            print("=" * 60)
    elif enable_continuum and not is_h2o_task:
        print("\n>> 提示: 检测到非水汽分子，跳过连续吸收计算。")

    # ---------------------------------------------------
    # F. HAPI 基准计算与缓存
    # ---------------------------------------------------
    hapi_results = None
    if skip_hapi:
        print("\n⚠️  跳过HAPI吸收截面光谱计算")
    else:
        # 如果 global_iso_ids 为 None，记录为 'default'，否则将列表转为字符串
        iso_tag = (
            "_".join(map(str, sorted(global_iso_ids))) if global_iso_ids else "default"
        )
        # 缓存键生成：基于分子、范围以及输入数据的特征(避免过长文件名)
        # 使用数据摘要(长度 + 首元素)来区分不同输入
        data_signature = (
            f"L{len(p_data)}_P{p_data[0]:.2f}_T{t_data[0]:.2f}_VMR{vmr_data[0]:.4f}"
        )
        cache_key = (
            f"{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{GLOBAL_WN_STEP}_{iso_tag}_{data_signature}"
        )
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        hapi_cache_path = os.path.join("cache", f"hapi_results_{cache_hash}.pkl")
        os.makedirs("cache", exist_ok=True)

        if os.path.exists(hapi_cache_path):
            print(f"\n发现HAPI吸收截面缓存数据: {hapi_cache_path}")
            print(f">> 缓存标识 (Isotopes): {iso_tag}")
            t_load_start = time.perf_counter()
            with open(hapi_cache_path, "rb") as f:
                hapi_results = pickle.load(f)
            print(f"缓存加载完成，耗时: {time.perf_counter() - t_load_start:.2f}秒")
        else:
            print("\n未找到缓存，开始HAPI计算...")
            t_hapi_start = time.perf_counter()
            hapi_results = Parallel(n_jobs=1)(
                delayed(calculate_hapi_benchmark_new)(
                    MOLECULE,
                    global_wavenumber_grid,
                    layer["temperature_k"],
                    layer["pressure_pa"],
                    GLOBAL_WN_MIN,
                    GLOBAL_WN_MAX,
                    global_iso_ids=global_iso_ids,
                    vmr=layer["vmr"],
                )
                for layer in tqdm(atmospheric_profile, desc="HAPI计算")
            )
            print(f"HAPI计算完成，耗时: {time.perf_counter() - t_hapi_start:.2f}秒")

            print("保存HAPI缓存...")
            with open(hapi_cache_path, "wb") as f:
                pickle.dump(hapi_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 无论结果是刚算出来的，还是缓存读出来的，都要在这里叠加
        if hapi_results is not None and should_run_continuum and continuum_cache:
            print("正在将连续吸收叠加到 HAPI 基准结果中...")
            # hapi_results 应该是一个列表，长度等于层数
            for i in range(len(hapi_results)):
                # 确保维度一致
                hapi_results[i] = hapi_results[i] + continuum_cache[i]
            print(">> HAPI 基准结果修正完成 (Voigt + MT_CKD)")

        else:
            print("连续吸收没加到hapi结果上")
    # ---------------------------------------------------
    # G. 数据保存
    # ---------------------------------------------------
    save_to_hdf5(
        output_path,
        atmospheric_profile,
        all_layer_absorptions,
        hapi_results,
        global_wavenumber_grid,
        MOLECULE,
    )

    print("=" * 80)
    print("全部完成！")
    print("=" * 80)


# ==============================================================================
# 3. 脚本入口 (User Interface)
#    这里处理用户输入：读取哪些文件，还是直接给数值，保存到哪里
# ==============================================================================
if __name__ == "__main__":

    # --- 1. 定义用户输入 (User Inputs) ---
    # 这些是每次运行可能变化的参数
    MOLECULE = "CO2"

    GLOBAL_WN_MIN = 600
    GLOBAL_WN_MAX = 700
    GLOBAL_WN_STEP = 0.01

    # 示例 B: 直接定义数值 (单层模式 - 如果你想用这个，注释掉上面的文件读取)
    target_p = 101325.0
    target_t = 296.0

    # 示例 A: 从文件读取 (大气廓线模式)
    # PRESSURE_FILE = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/pres_100.txt"
    # TEMPERATURE_FILE = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/US_STANDARD_ATMOSPHERE_T.txt"

    OUTPUT_H5 = f"../sigma_output_filefold/{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{GLOBAL_WN_STEP}_{target_p}_{target_t}.h5"

    # 示例 B: 直接定义数值 (单层模式 - 如果你想用这个，注释掉上面的文件读取)
    # target_p = 101325.0
    # target_t = 296.0

    # --- 2. 解析命令行标志 ---
    SKIP_HAPI_FLAG = "--skip-hapi" in sys.argv

    # --- 3. 准备数据 ---
    print("正在读取源数据文件...")
    # 注意：此处进行了单位转换 (mb -> Pa)，这是数据准备的一部分
    # input_p_vals = np.loadtxt(PRESSURE_FILE) * 100
    # input_t_vals = np.loadtxt(TEMPERATURE_FILE)

    # 如果是单层模式，这里可以是:
    input_p_vals = target_p
    input_t_vals = target_t

    # --- 4. 调用主函数 ---
    NNLBL_main(
        MOLECULE,
        GLOBAL_WN_MIN,
        GLOBAL_WN_MAX,
        GLOBAL_WN_STEP,
        input_pressures=input_p_vals,
        input_temperatures=input_t_vals,
        output_path=OUTPUT_H5,
        skip_hapi=SKIP_HAPI_FLAG,
        HP_MODEL_PATH="../NNmodel&stats/voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth",
        HP_STATS_PATH="../NNmodel&stats/voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy",
        LP_MODEL_PATH="../NNmodel&stats/voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth",
        LP_STATS_PATH="../NNmodel&stats/voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy",
    )
