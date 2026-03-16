"""
JQSRT期刊规范绘图工具
包含三种对比图风格：
  - create_jqsrt_comparison_figure          : 经典版（误差用MAE/RMSE/Max标注）
  - create_jqsrt_comparison_figure_relative_error     : 相对误差版（最大误差标注）
  - create_jqsrt_comparison_figure_relative_error_avg : 相对误差版（均值标注，推荐）
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# 样式辅助
# ---------------------------------------------------------------------------

def setup_jqsrt_plot_style():
    """设置符合JQSRT期刊规范的matplotlib样式（适配小四号字体文章）"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "mathtext.fontset": "stix",
        }
    )


# ---------------------------------------------------------------------------
# 内部布局常量
# ---------------------------------------------------------------------------

_MOLECULE_LATEX = {
    "O3": r"O$_3$",
    "H2O": r"H$_2$O",
    "CO2": r"CO$_2$",
    "O2": r"O$_2$",
}

# 4个气体在5行2列GridSpec中的位置 (主图行, 误差图行, 列)
_SUBPLOT_POSITIONS = [
    (0, 1, 0),  # O3  左列
    (0, 1, 1),  # H2O 右列
    (3, 4, 0),  # CO2 左列
    (3, 4, 1),  # O2  右列
]


def _make_gridspec(fig, height_ratios, hspace=0.12, wspace=0.24):
    return GridSpec(
        5, 2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace,
        left=0.08, right=0.97,
        top=0.95, bottom=0.06,
    )


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ 图像已保存: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 版本1：经典版（MAE / RMSE / Max 标注）
# ---------------------------------------------------------------------------

def create_jqsrt_comparison_figure(results_list, pressure_label, save_path):
    """创建4×2子图对比图（主图+误差图，误差用MAE/RMSE/Max标注）"""
    setup_jqsrt_plot_style()

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(
        5, 2, figure=fig,
        height_ratios=[3, 1.0, 0.7, 3, 1.0],
        hspace=0.15, wspace=0.25,
        left=0.08, right=0.97, top=0.94, bottom=0.05,
    )

    color_hapi = "#000000"
    color_model = "#E74C3C"
    color_error_zero = "#3498DB"
    color_error_line = "#7F8C8D"

    for idx, result in enumerate(results_list):
        if result is None:
            continue

        wn = result["wavenumber"]
        hapi_abs = result["hapi_absorption"]
        model_abs = result["model_absorption"]
        molecule = result["molecule"]
        wn_min, wn_max = result["wn_min"], result["wn_max"]

        main_row, error_row, col = _SUBPLOT_POSITIONS[idx]
        ax_main = fig.add_subplot(gs[main_row, col])
        ax_error = fig.add_subplot(gs[error_row, col])

        ax_main.plot(wn, hapi_abs, color=color_hapi, lw=0.9,
                     label="HAPI" if idx == 0 else "", alpha=0.95)
        ax_main.plot(wn, model_abs, color=color_model, lw=0.7,
                     label="NN Model" if idx == 0 else "", alpha=0.85)

        ax_main.set_yscale("log")
        valid_hapi = hapi_abs[hapi_abs > 0]
        if len(valid_hapi) > 0:
            ax_main.set_ylim(np.min(valid_hapi) * 0.3, np.max(hapi_abs) * 3.0)

        ax_main.set_xticklabels([])
        ax_error.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=9)
        ax_main.set_ylabel("Absorption Coeff.\n(cm$^2$/molecule)", fontsize=9, labelpad=2)

        label_letter_main = chr(97 + idx * 2)
        ax_main.text(0.02, 0.97, f"({label_letter_main})",
                     transform=ax_main.transAxes, fontsize=9, fontweight="bold",
                     va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor="gray", alpha=0.9, linewidth=0.6))

        title_text = (f"{_MOLECULE_LATEX.get(molecule, molecule)}: "
                      f"{wn_min}–{wn_max} cm$^{{-1}}$")
        ax_main.set_title(title_text, fontsize=10, pad=8)

        if idx == 0:
            ax_main.legend(loc="upper right", frameon=True,
                           framealpha=0.95, edgecolor="gray", fancybox=False)

        ax_main.grid(True, which="both", linestyle=":", lw=0.4, alpha=0.25)
        ax_main.minorticks_on()

        with np.errstate(divide="ignore", invalid="ignore"):
            relative_error = 100 * (model_abs - hapi_abs) / np.max(hapi_abs)

        ax_error.plot(wn, relative_error, color=color_error_line, lw=0.6, alpha=0.75)
        ax_error.axhline(0, color=color_error_zero, linestyle="--", lw=0.8, alpha=0.7)
        ax_error.set_ylim(-1, 1)
        ax_error.set_ylabel("Rel. Error (%)", fontsize=9, labelpad=2)

        label_letter_error = chr(97 + idx * 2 + 1)
        ax_error.text(0.02, 0.92, f"({label_letter_error})",
                      transform=ax_error.transAxes, fontsize=9, fontweight="bold",
                      va="top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="gray", alpha=0.9, linewidth=0.6))
        ax_error.grid(True, linestyle=":", lw=0.4, alpha=0.25)
        ax_error.minorticks_on()

        mae = np.mean(np.abs(relative_error))
        rmse = np.sqrt(np.mean(relative_error**2))
        max_err = np.max(np.abs(relative_error))
        stats_text = f"MAE={mae:.2f}%\nRMSE={rmse:.2f}%\nMax={max_err:.2f}%"
        ax_error.text(0.98, 0.96, stats_text, transform=ax_error.transAxes,
                      fontsize=6.5, va="top", ha="right", linespacing=1.4,
                      bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8DC",
                                edgecolor="#CD853F", alpha=0.95, linewidth=0.5))

        ax_error.set_xlim(ax_main.get_xlim())

    suptitle = (f'{pressure_label} Conditions '
                f'(P={results_list[0]["pressure"]:.0f} Pa, '
                f'T={results_list[0]["temperature"]:.1f} K)')
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.997)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# 版本2：相对误差版（最大误差标注）
# ---------------------------------------------------------------------------

def create_jqsrt_comparison_figure_relative_error(results_list, save_path):
    """创建4×2子图对比图（相对误差版，最大误差标注）"""
    STYLE = {
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.0, "figure.figsize": (10, 7),
    }
    plt.rcParams.update(STYLE)

    fig = plt.figure(figsize=STYLE["figure.figsize"])
    gs = _make_gridspec(fig, height_ratios=[3, 1, 0.9, 3, 1])

    color_hapi = "black"
    color_model = "#D62728"
    color_err_line = "#7F8C8D"
    color_err_zero = "#3498DB"

    for idx, result in enumerate(results_list):
        if result is None:
            continue

        wn = result["wavenumber"]
        hapi_abs = result["hapi_absorption"]
        model_abs = result["model_absorption"]
        molecule = result["molecule"]
        wn_min, wn_max = result["wn_min"], result["wn_max"]

        main_row, error_row, col = _SUBPLOT_POSITIONS[idx]
        ax_main = fig.add_subplot(gs[main_row, col])
        ax_error = fig.add_subplot(gs[error_row, col])

        ax_main.plot(wn, hapi_abs, color=color_hapi, lw=1.2, alpha=0.9,
                     label="Ground Truth" if idx == 0 else "")
        ax_main.plot(wn, model_abs, color=color_model, lw=1.2, alpha=0.9,
                     ls="--", dashes=(3, 1.5), label="NN Model" if idx == 0 else "")

        ax_main.set_yscale("log")
        valid_data = hapi_abs[hapi_abs > 1e-30]
        if len(valid_data) > 0:
            ax_main.set_ylim(np.min(valid_data) * 0.5, np.max(valid_data) * 5.0)

        ax_main.set_xticklabels([])
        ax_main.set_ylabel(r"$\sigma$ (cm$^2$/molecule)")

        label_letter = chr(97 + idx)
        title_text = (f"({label_letter}) {_MOLECULE_LATEX.get(molecule, molecule)}: "
                      f"{wn_min}–{wn_max} cm$^{{-1}}$")
        ax_main.set_title(title_text, loc="left", fontsize=11)

        if idx == 0:
            ax_main.legend(loc="upper left", frameon=False)

        rel_err_a = 100 * (model_abs - hapi_abs) / hapi_abs
        rel_err_b = 100 * (model_abs - hapi_abs) / np.max(hapi_abs)
        max_rel_val_a = np.max(np.abs(rel_err_a))
        max_rel_val_b = np.max(np.abs(rel_err_b))

        ax_error.plot(wn, rel_err_a, color=color_err_line, lw=0.8, alpha=0.8)
        ax_error.plot(wn, rel_err_b, color="r", lw=0.8, alpha=0.8)
        ax_error.axhline(0, color=color_err_zero, ls="--", lw=1.0, alpha=0.8)

        ax_error.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax_error.set_ylabel("RE (%)")
        ax_error.set_xlim(ax_main.get_xlim())

        limit = 2.0
        if max_rel_val_a > 0:
            limit = min(max(max_rel_val_a * 1.1, 0.1), 5)
        ax_error.set_ylim(-limit, limit)
        ax_error.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_error.yaxis.set_major_formatter(
            FormatStrFormatter("%.0e") if limit < 0.1 else FormatStrFormatter("%.1f")
        )

        ax_error.text(0.98, 0.90, f"Max Rel. Err = {max_rel_val_b:.3f}%",
                      transform=ax_error.transAxes, fontsize=9, c="r",
                      va="top", ha="right",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="gray", alpha=0.0))

        for ax in [ax_main, ax_error]:
            ax.grid(True, which="major", ls="-", lw=0.5, color="gray", alpha=0.3)
            ax.minorticks_on()

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# 版本3：相对误差版（均值标注，推荐）
# ---------------------------------------------------------------------------

def create_jqsrt_comparison_figure_relative_error_avg(results_list, save_path):
    """
    创建4×2子图对比图（相对误差版，在主图上标注两种误差计算方法的均值，推荐）
    - 方法A (灰色线): RE = (Model - Truth) / Truth × 100%
    - 方法B (红色线): Normalized RE = (Model - Truth) / max(Truth) × 100%
    """
    STYLE = {
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.0, "figure.figsize": (10, 7),
    }
    plt.rcParams.update(STYLE)

    fig = plt.figure(figsize=STYLE["figure.figsize"])
    gs = _make_gridspec(fig, height_ratios=[3, 1, 0.9, 3, 1])

    color_hapi = "black"
    color_model = "#D62728"
    color_err_line = "#7F8C8D"
    color_err_zero = "#3498DB"

    for idx, result in enumerate(results_list):
        if result is None:
            continue

        wn = result["wavenumber"]
        hapi_abs = result["hapi_absorption"]
        model_abs = result["model_absorption"]
        molecule = result["molecule"]
        wn_min, wn_max = result["wn_min"], result["wn_max"]

        main_row, error_row, col = _SUBPLOT_POSITIONS[idx]
        ax_main = fig.add_subplot(gs[main_row, col])
        ax_error = fig.add_subplot(gs[error_row, col])

        # 主图
        ax_main.plot(wn, hapi_abs, color=color_hapi, lw=1.2, alpha=0.9,
                     label="HAPI" if idx == 0 else "")
        ax_main.plot(wn, model_abs, color=color_model, lw=1.2, alpha=0.9,
                     ls="--", dashes=(3, 1.5), label="NN Model" if idx == 0 else "")

        ax_main.set_yscale("log")
        valid_data = hapi_abs[hapi_abs > 1e-30]
        if len(valid_data) > 0:
            ax_main.set_ylim(np.min(valid_data) * 0.5, np.max(valid_data) * 5.0)

        ax_main.set_xticklabels([])
        ax_main.set_ylabel(r"$\sigma$ (cm$^2$/molecule)")

        label_letter = chr(97 + idx)
        title_text = (f"({label_letter}) {_MOLECULE_LATEX.get(molecule, molecule)}: "
                      f"{wn_min}–{wn_max} cm$^{{-1}}$")
        ax_main.set_title(title_text, loc="left", fontsize=11)

        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color=color_hapi, lw=1.2, label="HAPI"),
                Line2D([0], [0], color=color_model, lw=1.2, ls="--",
                       dashes=(3, 1.5), label="NN Model"),
            ]
            ax_main.legend(handles=legend_elements, loc="upper left",
                           frameon=False, ncol=2, bbox_to_anchor=(0.02, 0.98))

        # 误差计算
        rel_err_a = 100 * (model_abs - hapi_abs) / hapi_abs
        rel_err_b = 100 * (model_abs - hapi_abs) / np.max(hapi_abs)
        avg_abs_a = np.mean(np.abs(rel_err_a))
        avg_abs_b = np.mean(np.abs(rel_err_b))

        ax_error.plot(wn, rel_err_a, color=color_err_line, lw=0.8, alpha=0.8)
        ax_error.plot(wn, rel_err_b, color="r", lw=0.8, alpha=0.8)
        ax_error.axhline(0, color=color_err_zero, ls="--", lw=1.0, alpha=0.8)

        ax_error.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax_error.set_ylabel("RE (%)")
        ax_error.set_xlim(ax_main.get_xlim())

        max_rel_val_a = np.max(np.abs(rel_err_a))
        if idx == 0:
            limit = 3.0
        else:
            limit = 2.0
            if max_rel_val_a > 0:
                limit = min(max(max_rel_val_a * 1.1, 0.1), 5)
        ax_error.set_ylim(-limit, limit)
        ax_error.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_error.yaxis.set_major_formatter(
            FormatStrFormatter("%.0e") if limit < 0.1 else FormatStrFormatter("%.1f")
        )

        # 第一张误差图添加图例
        if idx == 0:
            error_legend_elements = [
                Line2D([0], [0], color=color_err_line, lw=1.0, label="RE"),
                Line2D([0], [0], color="r", lw=1.0, label="Normalized RE"),
            ]
            ax_error.legend(handles=error_legend_elements, loc="upper left",
                            frameon=False, ncol=2, fontsize=8,
                            bbox_to_anchor=(0.02, 0.98))

        # 在主图上标注均值误差
        stats_text_a = f"Mean |RE| = {avg_abs_a:.4f}%"
        stats_text_b = f"Mean |RE|(norm) = {avg_abs_b:.4f}%"
        h_pos = 0.05 if molecule == "H2O" else 0.98
        ha = "left" if molecule == "H2O" else "right"

        for text, color in [(stats_text_a, color_err_line), (stats_text_b, "r")]:
            y_pos = 0.20 if text == stats_text_a else 0.10
            ax_main.text(h_pos, y_pos, text, transform=ax_main.transAxes,
                         fontsize=8, c=color, va="top", ha=ha,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                   edgecolor="gray", alpha=0.0))

        for ax in [ax_main, ax_error]:
            ax.grid(True, which="major", ls="-", lw=0.5, color="gray", alpha=0.3)
            ax.minorticks_on()

    _save(fig, save_path)
