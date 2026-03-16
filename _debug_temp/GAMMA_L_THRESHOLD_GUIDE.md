# NNLBL 洛伦兹半高半宽阈值模式使用指南

## 📌 更新概述

根据审稿人建议，NNLBL现在支持使用**洛伦兹半高半宽（Lorentz HWHM, γ_L）**作为HP/LP模型的划分依据。

### 为什么使用 γ_L 而不是气压？

- **物理意义更直接**：γ_L 直接反映谱线加宽状态
  - 高压环境：γ_L 大 → 压力加宽主导 → 谱线宽
  - 低压环境：γ_L 小 → 多普勒加宽主导 → 谱线窄

- **更准确的模型选择**：γ_L 同时考虑了压力和温度的影响，而单纯的气压阈值可能在某些温度条件下不够准确

---

## 🔧 如何使用

### 1. 配置文件设置

在 `example_config_NNLBL.py` 中设置 `GAMMA_L_THRESHOLD`：

```python
# 新模式：使用洛伦兹半高半宽阈值
GAMMA_L_THRESHOLD = 0.05  # 单位: cm⁻¹

# 旧模式：使用传统气压阈值（向后兼容）
GAMMA_L_THRESHOLD = None  # 将使用2200 Pa作为气压阈值
```

### 2. 如何确定阈值？

**最佳方法**：使用你的HP和LP模型训练时的 γ_L 边界值

例如，如果你的模型训练时：
- HP模型：γ_L ∈ [0.05, 0.5] cm⁻¹
- LP模型：γ_L ∈ [0.001, 0.05] cm⁻¹

则应设置：
```python
GAMMA_L_THRESHOLD = 0.05  # 边界值
```

**如果不确定边界**：
1. 可以先运行代码，查看输出的谱线参数分布
2. 或者根据典型大气条件下的 γ_L 值进行估算
3. 常见参考值：0.03 ~ 0.08 cm⁻¹

---

## 🆕 新模式 vs 旧模式对比

### 旧模式（气压阈值）

```python
GAMMA_L_THRESHOLD = None
```

**特点**：
- 整层使用同一个模型
- 按层划分：P ≥ 2200 Pa → HP，P < 2200 Pa → LP
- 计算速度较快
- 向后兼容旧版本

**适用场景**：快速测试、不关注极致精度

---

### 新模式（γ_L 阈值）

```python
GAMMA_L_THRESHOLD = 0.05  # 具体数值
```

**特点**：
- 同一层可以混合使用HP和LP模型
- 按谱线划分：每条谱线根据自己的 γ_L 选择模型
  - γ_L ≥ 阈值 → HP模型
  - γ_L < 阈值 → LP模型
- 物理上更合理
- 计算时间略有增加（因为每层需要两次推理）

**适用场景**：
- 审稿人要求
- 需要最高精度
- 过渡区域（气压~2200 Pa）的计算

---

## 📊 代码架构变化

### 核心修改点

#### 1. `pack_layers_into_batch` 函数（run_inference_and_save.py:327）

新增参数：
- `gamma_l_threshold`: γ_L 阈值
- `use_high_gamma`: True=选择HP谱线，False=选择LP谱线

功能：根据 γ_L 过滤谱线

#### 2. `NNLBL_main` 函数（NNLBL_main.py:386）

新增参数：
- `gamma_l_threshold`: 传递阈值

推理流程：
```python
if gamma_l_threshold is not None:
    # 新模式：遍历所有层
    for each_layer:
        hp_result = process_with_hp_model(gamma_l >= threshold)
        lp_result = process_with_lp_model(gamma_l < threshold)
        combined_result = hp_result + lp_result
else:
    # 旧模式：按气压分层
    hp_layers = layers with P >= 2200 Pa
    lp_layers = layers with P < 2200 Pa
```

---

## 💡 使用示例

### 示例1：启用 γ_L 阈值模式

```python
# example_config_NNLBL.py

TARGET_ISO_LIST = [1, 2]  # H2O的两个主要同位素
ENABLE_CONTINUUM = True

# 关键设置
GAMMA_L_THRESHOLD = 0.05  # 使用你的模型训练边界值

SPECTRAL_CONFIG = {
    "min": 4800.0,
    "max": 5200.0,
    "step": 0.01,
}

RUN_MODE = "PROFILE"
PROFILE_PARAMS = {
    "dir": "atmospheric_profile_for_testing",
    "p_file": "pres_100.txt",
    "p_unit": "hPa",
    "t_file": "US_STANDARD_ATMOSPHERE_T.txt",
    "t_unit": "K",
    "vmr_file": "US_STANDARD_ATMOSPHERE_h2o.txt",
    "vmr_unit": "ppmv",
    "name_tag": "US_STD_100",
}
```

运行：
```bash
python example_config_NNLBL.py
```

输出示例：
```
>> 模型划分模式: 洛伦兹半高半宽阈值 = 0.050000 cm⁻¹
   - γ_L ≥ 阈值 → HP模型（高压/宽谱线）
   - γ_L < 阈值 → LP模型（低压/窄谱线）

开始GPU推理（基于γ_L阈值的混合模型模式）...
每层将根据谱线的γ_L分别使用HP和LP模型
```

---

### 示例2：使用旧模式（向后兼容）

```python
# 保持旧行为
GAMMA_L_THRESHOLD = None
```

输出示例：
```
>> 模型划分模式: 气压阈值 = 2200.0 Pa（传统模式）
   - P ≥ 阈值 → HP模型
   - P < 阈值 → LP模型

开始GPU推理（传统气压阈值模式）...
```

---

## ⚠️ 注意事项

### 1. 计算时间
- 新模式下，每层需要调用两次模型（HP和LP），计算时间约为旧模式的1.5-2倍
- 但物理上更合理，适合发表论文

### 2. 阈值选择的重要性
- 阈值选择不当会导致模型性能下降
- 建议使用训练时的边界值
- 可以通过对比HAPI结果来验证阈值的合理性

### 3. 边界情况
- 如果某层所有谱线的 γ_L 都在阈值同一侧，行为与旧模式相同
- 如果某层没有谱线（罕见），会返回零吸收

### 4. 调试技巧
如果想查看每层有多少谱线被分配给HP/LP，可以在代码中添加打印：

```python
# 在 NNLBL_main.py 的新模式循环中添加：
gamma_l_values = all_layers_lines_params[idx]["gamma_l"]
n_hp = np.sum(gamma_l_values >= gamma_l_threshold)
n_lp = np.sum(gamma_l_values < gamma_l_threshold)
print(f"Layer {idx}: {n_hp} HP lines, {n_lp} LP lines")
```

---

## 🔬 技术细节

### γ_L 的计算

在 `run_inference_and_save.py:305` 中，通过HAPI的 `calculateProfileParametersVoigt` 计算：

```python
line_calc_params = calculateProfileParametersVoigt(TRANS=trans)
lines_params["gamma_l"][i] = line_calc_params["Gamma0"]
```

γ_L 的物理含义：
- 洛伦兹线型的半高半宽（HWHM）
- 单位：cm⁻¹
- 与压力成正比，与温度相关

### 谱线过滤逻辑

在 `pack_layers_into_batch` 中：

```python
if gamma_l_threshold is not None:
    gamma_l_arr = params["gamma_l"]
    if use_high_gamma:
        mask = gamma_l_arr >= gamma_l_threshold  # HP模型
    else:
        mask = gamma_l_arr < gamma_l_threshold   # LP模型

    # 应用mask过滤所有参数
    filtered_params = {k: params[k][mask] for k in params}
```

---

## 📝 总结

✅ **已完成的修改**：
1. 添加了基于 γ_L 的谱线过滤功能
2. 支持同一层混合使用HP和LP模型
3. 保持向后兼容（可切换回旧模式）
4. 添加了清晰的配置接口

🎯 **使用建议**：
- 论文发表：使用 γ_L 阈值模式（更符合物理原理）
- 快速测试：使用气压阈值模式（计算更快）
- 首次使用：先用 `GAMMA_L_THRESHOLD = None` 测试，确保代码运行正常

📧 **如有问题**：
- 检查阈值设置是否合理
- 对比新旧模式的结果差异
- 使用HAPI结果验证精度
