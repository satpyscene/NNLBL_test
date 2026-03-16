import sys
from pathlib import Path
from NNLBL_src.NNLBL_main import NNLBL_API, validate_user_config

# ------------------------------------------------------------------------------
# 1. Target isotopes (HITRAN global isotope IDs)
#    ⚠️ 只能来自同一个分子 (e.g. 全是 H2O 或全是 CO2)
# ------------------------------------------------------------------------------
# H2O: [1, 2, 3, 4, 5, 6, 129]
# CO2: [7, 8, 9, 10, 11, 12, 13, 14]
# O3 : [16, 17, 18, 19, 20]
# N2O: [21, 22, 23, 24, 25]
# CO : [26, 27, 28, 29, 30, 31]
# CH4: [32, 33, 34, 35]
# O2 : [36, 37, 38]

TARGET_ISO_LIST = [7]  # 示例: H2O 的 1、2 号同位素
ENABLE_CONTINUUM = False  # 是否启用 MT-CKD 连续吸收 (仅对 H2O 有意义)

# ------------------------------------------------------------------------------
# 1.5 模型划分策略（重要！）
# ------------------------------------------------------------------------------
# 设置洛伦兹半高半宽阈值，用于决定谱线使用HP还是LP模型
#
# 使用建议：
#   - 依据HP和LP模型训练时的gamma_l边界，设置该值
#   - 例如：GAMMA_L_THRESHOLD = 0.001  # 单位: cm⁻¹
#   - 设置为 None 则使用传统的气压阈值模式（2200 Pa）
#
# 物理意义：
#   - gamma_l 较大 → 压力加宽主导 → 谱线宽 → 使用HP模型
#   - gamma_l 较小 → 多普勒加宽主导 → 谱线窄 → 使用LP模型
#
GAMMA_L_THRESHOLD = 0.001  # 设置为具体数值（如0.001）启用新模式，None使用旧模式

# ------------------------------------------------------------------------------
# 2. Spectral configuration
#    单位: wavenumber [cm⁻¹]
# ------------------------------------------------------------------------------
SPECTRAL_CONFIG = {
    "min": 4800.0,  # 起始波数 [cm^-1]
    "max": 5200.0,  # 终止波数 [cm^-1]
    "step": 0.01,  # 步长   [cm^-1]
}

# ------------------------------------------------------------------------------
# 3. Run mode
#    "SINGLE"  : 单层均匀大气
#    "PROFILE" : 垂直廓线大气
# ------------------------------------------------------------------------------
RUN_MODE = "SINGLE"


# ------------------------------------------------------------------------------
# 4A. SINGLE-layer atmospheric state
#    所有单位必须显式写出，否则程序会直接报错
# ------------------------------------------------------------------------------
SINGLE_PARAMS = {
    "p_hpa": 1013.250,  # Pressure    [hPa]
    "t_k": 280.0,  # Temperature [K]
    "vmr_ppmv": 0.0,  # Volume mixing ratio [ppmv] (40000 ppmv = 4%)
}


# ------------------------------------------------------------------------------
# 4B. PROFILE atmospheric state (vertical profile)
# ------------------------------------------------------------------------------
# 所有文件均为一维数组，长度必须一致
# 单位必须通过 *_unit 字段显式声明
PROFILE_PARAMS = {
    "dir": "atmospheric_profile_for_testing",
    # Pressure profile
    "p_file": "pres_100.txt",
    "p_unit": "hPa",  # 可选: "Pa", "hPa"
    # Temperature profile
    "t_file": "US_STANDARD_ATMOSPHERE_T.txt",
    "t_unit": "K",  # 可选: "K", "C"
    # Gas volume mixing ratio profile (optional)
    "vmr_file": "US_STANDARD_ATMOSPHERE_h2o.txt",
    "vmr_unit": "ppmv",  # 可选: "vmr", "ppmv"
    "name_tag": "US_STD_100",
}

# ------------------------------------------------------------------------------
# 5. Paths (normally no need to change)
# ------------------------------------------------------------------------------
PATH_CONFIG = {
    "base_dir": Path(__file__).parent,
    "model_dir": "NNmodel&stats",
    "output_dir": "sigma_output_filefold",
    "mtckd_file": "data/absco-ref_wv-mt-ckd.nc",
}


# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":

    # ---------- 用户输入前置校验 ----------
    validate_user_config(
        run_mode=RUN_MODE,
        single_params=SINGLE_PARAMS,
        profile_params=PROFILE_PARAMS,
        spectral_config=SPECTRAL_CONFIG,
        target_iso_list=TARGET_ISO_LIST,
    )

    NNLBL_API(
        target_iso_list=TARGET_ISO_LIST,
        spectral_config=SPECTRAL_CONFIG,
        input_mode=RUN_MODE,
        single_config=SINGLE_PARAMS,
        profile_config=PROFILE_PARAMS,
        path_config=PATH_CONFIG,
        enable_continuum=ENABLE_CONTINUUM,
        skip_hapi=("--skip-hapi" in sys.argv),
        gamma_l_threshold=GAMMA_L_THRESHOLD,  # 新增：传递gamma_l阈值
    )
