import numpy as np
import netCDF4 as nc
import os


class MTCKD_H2O:
    """
    MT-CKD 水汽连续吸收模型 (Python版)
    对应 Fortran 模块: mt_ckd_h2o, read_file, phys_consts
    """

    # --- 来自 phys_consts 模块的常量 ---
    RADCN2 = 1.4387752  # cm K

    def __init__(self, nc_path, use_closure=False):
        """
        初始化模型，加载 NetCDF 数据。

        参数:
            nc_path (str): 'absco-ref_wv-mt-ckd.nc' 文件的路径
            use_closure (bool): 对应 Fortran 中的 FRGNX 参数。
                                True 读取 'for_closure_absco_ref' (FRGNX='1')
                                False 读取 'for_absco_ref' (默认)
        """
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"找不到数据文件: {nc_path}")

        self.nc_path = nc_path

        # --- 对应 read_file 模块的 getData 功能 ---
        with nc.Dataset(nc_path, "r") as ds:
            # 读取基准数据
            self.wvn_ref = ds.variables["wavenumbers"][:]
            self.self_absco_ref = ds.variables["self_absco_ref"][:]
            self.self_texp = ds.variables["self_texp"][:]
            self.ref_temp = float(ds.variables["ref_temp"][:])
            self.ref_press = float(ds.variables["ref_press"][:])

            # 处理 FRGNX 逻辑
            if use_closure:
                if "for_closure_absco_ref" in ds.variables:
                    self.for_absco_ref = ds.variables["for_closure_absco_ref"][:]
                    print("Info: Loaded foreign closure absorption coefficients.")
                else:
                    raise KeyError("NetCDF文件中不存在 'for_closure_absco_ref'")
            else:
                self.for_absco_ref = ds.variables["for_absco_ref"][:]

        # 预计算一些参数，方便后续使用
        self.dvc = self.wvn_ref[1] - self.wvn_ref[0]  # 基准网格步长
        self.min_wvn_ref = self.wvn_ref[0]
        self.max_wvn_ref = self.wvn_ref[-1]

    def _radiation_term(self, wvn_array, temperature):
        """
        对应 Fortran 的 myradfn 函数
        计算辐射项因子: nu * tanh(h*nu / 2kT)
        使用 Numpy 向量化替代了 Fortran 的 if/else 循环
        """
        xkt = temperature / self.RADCN2
        xviokt = wvn_array / xkt

        # 物理公式: nu * (1 - exp(-nu/kT)) / (1 + exp(-nu/kT))
        # 等价于 nu * tanh(nu / 2kT)
        # 这种写法数值稳定性好，不需要像 Fortran 那样手动分段处理极小值
        rad = wvn_array * np.tanh(0.5 * xviokt)

        return rad

    def _cubic_interpolation(
        self, coarse_val, idx_start, idx_end, target_wvn, fine_vals
    ):
        """
        对应 Fortran 的 myxint 子程序
        执行 4点三次卷积插值 (4-point Cubic Convolution Interpolation)

        参数:
            coarse_val: 基准数据数组 (如 self_absco_ref)
            idx_start, idx_end: 基准数据中需要使用的索引范围 (Python slice indices)
            target_wvn: 目标波数网格
            fine_vals: 预分配的结果数组 (将被原地修改)
        """
        # 提取相关区间的基准数据
        # 注意：为了插值，我们需要左边多取1个点，右边多取2个点，以此凑齐4个点
        # Fortran逻辑: J-1, J, J+1, J+2
        subset_wvn = self.wvn_ref[idx_start:idx_end]

        # 计算每个目标波数对应在基准网格中的浮点索引位置
        # position = (target - origin) / step
        positions = (target_wvn - self.min_wvn_ref) / self.dvc

        # 计算基准网格的整数索引 j (对应 Fortran 中的 J)
        # 这里的 j 是相对于整个 self.wvn_ref 的索引
        j_indices = np.floor(positions).astype(int)

        # 确保索引不越界 (虽然理论上 external logic 保证了范围，但为了稳健)
        j_indices = np.clip(j_indices, 1, len(self.wvn_ref) - 3)

        # 计算相对距离 p (0 <= p < 1)
        p = positions - j_indices

        # 计算卷积系数 (完全对应 Fortran 代码)
        # C = (3.-2.*P)*P*P
        # B = 0.5*P*(1.-P)
        c = (3.0 - 2.0 * p) * p**2
        b = 0.5 * p * (1.0 - p)
        b1 = b * (1.0 - p)
        b2 = b * p

        # 获取 4 个邻近点的值
        # Python索引 j 对应 Fortran 的 A(J)
        # Fortran: A(J-1), A(J), A(J+1), A(J+2)
        # Python:  arr[j-1], arr[j], arr[j+1], arr[j+2]
        a_j_minus_1 = self.wvn_ref  # 这里只是占位，实际应该取传入的 value array

        # 为了速度，我们直接用 numpy 的高级索引取值
        v_jm1 = coarse_val[j_indices - 1]
        v_j = coarse_val[j_indices]
        v_jp1 = coarse_val[j_indices + 1]
        v_jp2 = coarse_val[j_indices + 2]

        # 计算插值结果
        # CONTI = -A(J-1)*B1+A(J)*(1.-C+B2)+A(J+1)*(C+B1)-A(J+2)*B2
        conti = -v_jm1 * b1 + v_j * (1.0 - c + b2) + v_jp1 * (c + b1) - v_jp2 * b2

        # 存入结果 (对应 R3(I) = R3(I) + CONTI * AFACT)
        # 这里直接赋值，因为 Python 版本我们通常初始化为0或直接生成新数组
        fine_vals[:] = conti

    def get_absorption(self, p_atm, t_atm, h2o_vmr, wv1, wv2, dwv, radflag=True):
        """
        对应 Fortran 的 mt_ckd_h2o_absco 子程序

        参数:
            p_atm (float): 大气压强 (单位与 NetCDF ref_press 一致，通常为 atm 或 hPa，请注意输入值需匹配 ref_press 单位)
                           *注意*: Fortran代码中 rho_rat = p_atm/ref_press。
                           如果 ref_press 是 1013 (hPa)，你输入 p_atm 也应该是 hPa。
                           如果输入是 Pa，请先 /100.0 或 /101325.0 转换。
            t_atm (float): 大气温度 (K)
            h2o_vmr (float): 水汽体积分数 (无量纲, e.g., 0.01)
            wv1, wv2 (float): 计算波数范围 (cm-1)
            dwv (float): 输出波数步长 (cm-1)
            radflag (bool): 是否乘以辐射项 (默认 True)

        返回:
            tuple: (wvn_out, self_absco, for_absco)
                   - wvn_out: 输出波数网格
                   - self_absco: 自加宽吸收系数
                   - for_absco: 外加宽吸收系数
        """

        # 1. 确定基准数据的索引范围 (对应 Find coeff wavenumber range)
        # 稍微放宽范围 (+2*dvc) 以便插值
        mask_indices = np.where(
            (self.wvn_ref >= (wv1 - 2 * self.dvc))
            & (self.wvn_ref <= (wv2 + 2 * self.dvc))
        )[0]

        if len(mask_indices) == 0:
            return np.array([]), np.array([]), np.array([])

        i1, i2 = mask_indices[0], mask_indices[-1]

        # 提取该范围内的基准数据 (避免对整个大数组运算，节省内存/时间)
        # 注意: Python 切片是左闭右开，所以用 i2+1
        # 但插值时需要 i2+2 的数据，所以我们这里切宽一点
        i_end_safe = min(len(self.wvn_ref), i2 + 5)
        wvn_subset = self.wvn_ref[i1:i_end_safe]
        self_ref_subset = self.self_absco_ref[i1:i_end_safe]
        for_ref_subset = self.for_absco_ref[i1:i_end_safe]
        texp_subset = self.self_texp[i1:i_end_safe]

        # 2. 计算物理修正参数 (Define some atmospheric parameters)
        # rho_rat = (p / p_ref) * (T_ref / T)
        rho_rat = (p_atm / self.ref_press) * (self.ref_temp / t_atm)

        # 3. 计算 Self Continuum (粗网格)
        # C_self = C_ref * (T_ref/T)^N * VMR * rho_rat
        # 对应: sh2o_coeff = ...
        sh2o_coeff_coarse = self_ref_subset * (self.ref_temp / t_atm) ** texp_subset
        sh2o_coeff_coarse *= h2o_vmr * rho_rat

        # 4. 计算 Foreign Continuum (粗网格)
        # C_for = C_ref * (1-VMR) * rho_rat
        fh2o_coeff_coarse = for_ref_subset * ((1.0 - h2o_vmr) * rho_rat)

        # 5. 辐射项修正 (Radiation Term)
        if radflag:
            rad_term = self._radiation_term(wvn_subset, t_atm)
            sh2o_coeff_coarse *= rad_term
            fh2o_coeff_coarse *= rad_term

        # 6. 插值到目标网格 (Interpolate)
        # 生成输出网格
        num_points = int((wv2 - wv1) / dwv) + 1
        wvn_out = wv1 + np.arange(num_points) * dwv

        # 预分配输出数组
        self_absco_out = np.zeros(num_points)
        for_absco_out = np.zeros(num_points)

        # 调用插值函数
        # 这里我们传入整个 self.wvn_ref 对应的 values 比较麻烦
        # 我们直接使用前面切片好的 coarse 数组，但需要注意索引对齐
        # 简单的办法：直接用前面切出的 coarse 数组作为源，
        # 但 myxint 需要源数据的 X 轴信息。
        # 既然我们完全重写了插值，可以直接用全局索引逻辑。

        self._cubic_interpolation(self.self_absco_ref, 0, -1, wvn_out, self_absco_out)
        self._cubic_interpolation(self.for_absco_ref, 0, -1, wvn_out, for_absco_out)

        # 注意: 上面的插值是对 原始 ref 数据插值吗？
        # 不！Fortran 是先在粗网格上计算好物理修正后的值 (sh2o_coeff)，然后再插值。
        # 所以我们需要用 sh2o_coeff_coarse 作为源数据。
        # 但 sh2o_coeff_coarse 只是整个波段的一部分。
        # 为了简化逻辑，我们重新写一个局部的插值调用：

        # 定义局部插值函数 (适配切片后的数据)
        def interpolate_local(coarse_y, coarse_x_start_val, target_x, out_y):
            # 局部坐标转换
            positions = (target_x - coarse_x_start_val) / self.dvc
            j_indices = np.floor(positions).astype(int)
            # 确保不越界 (相对于切片数组)
            j_indices = np.clip(j_indices, 1, len(coarse_y) - 3)
            p = positions - j_indices

            c = (3.0 - 2.0 * p) * p**2
            b = 0.5 * p * (1.0 - p)
            b1, b2 = b * (1.0 - p), b * p

            v_jm1 = coarse_y[j_indices - 1]
            v_j = coarse_y[j_indices]
            v_jp1 = coarse_y[j_indices + 1]
            v_jp2 = coarse_y[j_indices + 2]

            out_y[:] = (
                -v_jm1 * b1 + v_j * (1.0 - c + b2) + v_jp1 * (c + b1) - v_jp2 * b2
            )

        # 执行插值
        interpolate_local(sh2o_coeff_coarse, wvn_subset[0], wvn_out, self_absco_out)
        interpolate_local(fh2o_coeff_coarse, wvn_subset[0], wvn_out, for_absco_out)

        return wvn_out, self_absco_out, for_absco_out


# --- 使用示例 (对应 program main) ---
if __name__ == "__main__":
    # 假设 nc 文件路径
    nc_file = "/Users/user/Desktop/0_NNLBL_main_use/data/absco-ref_wv-mt-ckd.nc"

    if not os.path.exists(nc_file):
        print(f"错误: 找不到文件 {nc_file}")
    else:
        # 1. 初始化模型
        model = MTCKD_H2O(nc_file)

        # 2. 你的测试配置 (来自 Namelist)
        p_atm_input = 1013.0  # hPa
        t_atm_input = 300.0  # K
        h2o_frac_input = 0.00990098  # VMR
        wv1_input = 497.0
        wv2_input = 603.0
        dwv_input = 1.0

        # --- 安全检查: 检查压力单位是否匹配 ---
        # 如果用户输入了 101325 (Pa) 但基准是 1013 (hPa)，比值会达到 100，这显然不对
        ratio_check = p_atm_input / model.ref_press
        if ratio_check > 10.0:
            print(
                f"⚠️ 警告: 输入气压 ({p_atm_input}) 远大于参考气压 ({model.ref_press})。"
            )
            print("请确认输入单位是 hPa/mbar 而不是 Pa？")

        # 3. 运行计算
        print(f"--- 开始计算 ---")
        print(
            f"范围: {wv1_input} - {wv2_input} cm-1, T={t_atm_input}K, P={p_atm_input}hPa"
        )

        nu, self_abs, for_abs = model.get_absorption(
            p_atm=p_atm_input,
            t_atm=t_atm_input,
            h2o_vmr=h2o_frac_input,
            wv1=wv1_input,
            wv2=wv2_input,
            dwv=dwv_input,
        )

        # 4. 验证输出
        expected_len = int((wv2_input - wv1_input) / dwv_input) + 1
        print(f"计算完成!")
        print(f"预期点数: {expected_len}")
        print(f"实际点数: {len(nu)}")

        if len(nu) == expected_len:
            print("✅ 数组长度校验通过 (一致)")
        else:
            print("❌ 数组长度不匹配，请检查 dwv 精度")

        # 打印前5个数据供对比 (你可以和Fortran输出对比)
        print("\n前5个波数的计算结果 (Self / Foreign):")
        for i in range(5):
            print(
                f"Wave: {nu[i]:.2f} | Self: {self_abs[i]:.4e} | For: {for_abs[i]:.4e}"
            )
