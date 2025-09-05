# -*- coding: utf-8 -*-
"""
B题·第二问（v2）：稳健求条纹间距 Δν 与厚度 t 的完整脚本
--------------------------------------------------------
改动要点：
1) **ROI 选择**：仅在条纹干净的波数段估计 Δν（默认 1700–4000 cm^-1）。
2) **两种厚度求解模式**（输出可同时保存）：
   - MODE='assume_n'：给定材料先验折射率 n（如 SiC≈2.6 / Si≈3.42），
     用公式 t = 1/(2 n cosθ' Δν) 分别对两个角求 t，再做稳健汇总（推荐）。
   - MODE='dual_angle'：按两角 Δν 的比值消去 t 反演 n(ν)，再回代 t（角度太近易病态，仅作备选）。
3) **滑窗FFT + 零填充** 提高 Δν 分辨率；
4) **SNR 筛选 + 块自举 CI**：剔除弱条纹窗口，考虑滑窗相关性。
5) **中文字体**：默认 SimHei，避免中文告警。

使用：
- 把 FILE_10/FILE_15 改成你的绝对路径；如需改材料 n，调整 MATERIAL_N。
- 运行：python Problem2_v2.py
- 输出：./outputs/ 下的图像与 result_v2.xlsx
"""

import os
import math
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
from numpy.random import default_rng

# =========================== 文件路径（请按需修改） ===========================
FILE_10 = 'D:/MathModeling/2025国赛B题/附件/附件1.xlsx'  # 10° 入射
FILE_15 = 'D:/MathModeling/2025国赛B题/附件/附件2.xlsx'  # 15° 入射
SHEET_10 = 0
SHEET_15 = 0

# =========================== 全局绘图字体 ===========================
plt.rcParams['font.family'] = 'SimHei'     # 中文
plt.rcParams['axes.unicode_minus'] = False # 负号

# =========================== 物理/算法参数 ===========================
THETA1_DEG = 10.0
THETA2_DEG = 15.0
N0_EXT = 1.0

# —— 选择求解模式：'assume_n'（推荐）或 'dual_angle' ——
MODE = 'assume_n'
MATERIAL_N = 2.60   # SiC≈2.6；若 Si 取 3.42；可按题设调整

# —— 若走 dual_angle，n 的搜索范围 ——
N_SEARCH_MIN, N_SEARCH_MAX = 1.05, 5.00

# —— 仅在 ROI 内估计 Δν（可配多段），默认避开 1000 cm^-1 附近强吸收 ——
ROIS = [(1600.0, 3800.0)]  # 可改成 [(1700, 2600), (2800, 4000)] 等

# —— 去基线参数 ——
BASELINE_FRAC = 0.06
BASELINE_POLY = 3

# —— 滑窗 FFT 参数（局部 Δν） ——
WIN_CM1 = 300.0   # 窗口宽度（越大越稳，但跨频段时会平滑过度）
STEP_CM1 = 60.0   # 步长
ZP_FACTOR = 8    # 零填充倍数（提高频率分辨率）

# —— 峰间距（全局粗估）参数 ——
PEAK_PROM_FACTOR = 0.35
MIN_PEAK_DIST_CM1 = 0.4

# —— SNR 过滤：主峰幅度 / 次峰幅度 比值阈值 ——
SNR_THRESH = 1.00

# —— 置信区间：块自举 ——
N_BOOT = 800
BLOCK = 5
RANDOM_SEED = 2025

# 输出目录
OUTDIR = Path('./outputs')
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================== 工具函数 ===========================

def read_spectrum_specific(path: str, sheet=0) -> Tuple[np.ndarray, np.ndarray]:
    """读取谱：要求列名 '波数 (cm-1)' 与 '反射率 (%)'；返回升序 (wn, R[0~1])。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'文件不存在：{p}')
    if p.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(p, sheet_name=sheet)
    elif p.suffix.lower() == '.csv':
        df = pd.read_csv(p)
    else:
        raise ValueError(f'不支持的文件类型：{p.suffix}')

    col_wn, col_Rp = '波数 (cm-1)', '反射率 (%)'
    if col_wn not in df.columns or col_Rp not in df.columns:
        raise ValueError(f"未找到所需列。实际列：{list(df.columns)}；期望 '{col_wn}', '{col_Rp}'")

    wn = df[col_wn].to_numpy(float)
    R = (df[col_Rp].to_numpy(float)) / 100.0

    m = np.isfinite(wn) & np.isfinite(R)
    wn, R = wn[m], R[m]
    order = np.argsort(wn)
    return wn[order], R[order]


def restrict_to_rois(wn: np.ndarray, y: np.ndarray, rois) -> Tuple[np.ndarray, np.ndarray]:
    """按 ROI 切片后拼接（保持单调升序）。"""
    parts_w, parts_y = [], []
    for lo, hi in rois:
        m = (wn >= lo) & (wn <= hi)
        parts_w.append(wn[m])
        parts_y.append(y[m])
    if not parts_w:
        return wn, y
    return np.concatenate(parts_w), np.concatenate(parts_y)


def to_uniform_grid(wn: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """线性插值到等步长波数栅格，方便 FFT/峰检测。"""
    N = len(wn)
    wn_eq = np.linspace(wn[0], wn[-1], N)
    f = interp1d(wn, R, kind='linear', fill_value='extrapolate', assume_sorted=True)
    return wn_eq, f(wn_eq)


def remove_baseline(y: np.ndarray, frac=0.06, poly=3) -> Tuple[np.ndarray, np.ndarray]:
    """Savitzky–Golay 去基线，返回 (baseline, y_ac)。"""
    N = len(y)
    win = max(11, int(N*frac)//2*2 + 1)
    baseline = savgol_filter(y, window_length=win, polyorder=poly, mode='interp')
    return baseline, y - baseline


def estimate_delta_nu_by_peaks(wn, y_ac, prom_factor=0.35, min_dist_cm1=0.4):
    """A) 全局峰间距法：返回 Δν(中位数), ~1σ, 所有间距。"""
    std_ac = float(np.std(y_ac))
    prom = max(1e-8, prom_factor * std_ac)
    dnu = (wn[-1] - wn[0]) / max(len(wn) - 1, 1)
    min_dist_pts = max(3, int(min_dist_cm1 / max(dnu, 1e-9)))
    peaks, _ = find_peaks(y_ac, prominence=prom, distance=min_dist_pts)
    wn_peaks = wn[peaks]
    if len(wn_peaks) < 3:
        return np.nan, np.nan, np.array([])
    deltas = np.diff(wn_peaks)
    med = float(np.median(deltas))
    q25, q75 = np.percentile(deltas, [25, 75])
    sigma = float((q75 - q25) / 1.349)
    return med, sigma, deltas


def local_delta_nu_fft(wn: np.ndarray, y_ac: np.ndarray,
                       win_cm1=300.0, step_cm1=60.0, zp=4,
                       snr_thresh=2.0):
    """B) 滑窗 FFT + 零填充：返回窗口中心、Δν、以及每窗 SNR。"""
    dnu = (wn[-1] - wn[0]) / max(len(wn) - 1, 1)
    win_pts = max(64, int(win_cm1 / max(dnu, 1e-9)))
    step_pts = max(8, int(step_cm1 / max(dnu, 1e-9)))

    centers, deltas, snrs = [], [], []
    i = 0
    while i + win_pts <= len(wn):
        seg = y_ac[i:i+win_pts]
        seg = seg - np.mean(seg)
        # 零填充提高频率分辨率
        pad = int(win_pts * max(1, zp))
        X = rfft(seg * np.hanning(win_pts), n=pad)
        freq = rfftfreq(pad, d=dnu)  # cycles/(cm^-1)
        if len(freq) <= 4:
            break
        # 忽略极低频，找主峰与次峰
        kmin = 1
        mag = np.abs(X)
        kmax = int(np.argmax(mag[kmin:])) + kmin
        main = mag[kmax]
        # 次峰：把主峰附近 ±1 个 bin 置零后再找
        tmp = mag.copy()
        tmp[max(kmin, kmax-1):kmax+2] = 0
        sec = float(tmp.max()) if tmp.size else 1e-12
        snr = float(main / max(sec, 1e-12))
        if freq[kmax] > 0:
            centers.append(0.5*(wn[i] + wn[i+win_pts-1]))
            deltas.append(1.0 / freq[kmax])
            snrs.append(snr)
        i += step_pts

    return np.array(centers), np.array(deltas), np.array(snrs)


def cos_theta_t(n: float, theta_ext_rad: float, n0=1.0) -> float:
    s = n0 * math.sin(theta_ext_rad) / max(n, 1e-12)
    if s >= 1.0:
        return float('nan')
    return math.sqrt(max(0.0, 1.0 - s*s))


def solve_n_from_ratio(rho_target: float, theta1: float, theta2: float,
                       n0=1.0, n_lo=1.05, n_hi=5.0) -> float:
    if not (np.isfinite(rho_target) and rho_target > 0):
        return np.nan
    lo, hi = n_lo, n_hi
    for _ in range(64):
        mid = 0.5*(lo+hi)
        c1 = cos_theta_t(mid, theta1, n0)
        c2 = cos_theta_t(mid, theta2, n0)
        if not (np.isfinite(c1) and np.isfinite(c2)):
            return np.nan
        rho_mid = c2 / max(c1, 1e-12)
        if rho_mid < rho_target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def robust_median_ci_block(x: np.ndarray, n_boot=800, block=5, seed=2025):
    """块自举的中位数与 95%CI。"""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, (np.nan, np.nan)
    med = float(np.median(x))
    if len(x) < 10 or n_boot <= 0:
        return med, (np.nan, np.nan)
    M = len(x)
    n_blocks = max(1, M // block)
    rng = default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idxs = []
        for _ in range(n_blocks):
            s = int(rng.integers(0, max(M-block+1, 1)))
            idxs.extend(range(s, min(s+block, M)))
        boots.append(np.median(x[np.array(idxs[:M])]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, (float(lo), float(hi))

# =========================== 主流程 ===========================

def process_one(path: str, sheet=0):
    """读取→ROI→等步长→去基线→Δν(峰法、滑窗FFT+SNR)。"""
    wn, R = read_spectrum_specific(path, sheet)
    wn, R = restrict_to_rois(wn, R, ROIS)
    wn_eq, R_eq = to_uniform_grid(wn, R)
    baseline, R_ac = remove_baseline(R_eq, BASELINE_FRAC, BASELINE_POLY)

    # 全局 Δν 粗估（峰法）
    dnu_med, dnu_sig, deltas = estimate_delta_nu_by_peaks(wn_eq, R_ac,
                                                          PEAK_PROM_FACTOR, MIN_PEAK_DIST_CM1)
    # 局部 Δν（带 SNR）
    centers, dnu_loc, snr = local_delta_nu_fft(wn_eq, R_ac, WIN_CM1, STEP_CM1, ZP_FACTOR, SNR_THRESH)
    # SNR 筛选
    m = snr >= SNR_THRESH
    if m.sum() < 5:  # 通过窗口少于5个则放宽
        print(f"[WARN] SNR≥{SNR_THRESH} 仅通过 {m.sum()} 个窗口，放宽阈值至 1.02")
        m = snr >= 1.02
    if m.sum() < 5:  # 仍然太少，则不筛
        print("[WARN] 放弃SNR筛选，使用所有窗口（仅本次）")
        m = np.ones_like(snr, dtype=bool)

    return {
        'wn': wn_eq, 'R': R_eq, 'baseline': baseline, 'R_ac': R_ac,
        'dnu_global': np.array([dnu_med, dnu_sig]), 'deltas_all': deltas,
        'centers': centers[m], 'dnu_local': dnu_loc[m], 'snr': snr[m]
    }


def thickness_from_dnu(delta_nu, n, theta_deg, n0=1.0):
    """由 Δν（cm^-1）和给定 n 求厚度（cm）。"""
    th = math.radians(theta_deg)
    ctp = cos_theta_t(n, th, n0)
    return 1.0 / (2.0 * n * max(ctp, 1e-12) * np.maximum(delta_nu, 1e-12))


def solve_dual_angle_and_export(file10: str, file15: str, sheet10=0, sheet15=0):
    print('[I] 10° 读取+处理...')
    d10 = process_one(file10, sheet10)
    print('[I] 15° 读取+处理...')
    d15 = process_one(file15, sheet15)

    print(f"[10°] 全局 Δν(峰法) ≈ {d10['dnu_global'][0]:.4f} ± {d10['dnu_global'][1]:.4f} cm^-1")
    print(f"[15°] 全局 Δν(峰法) ≈ {d15['dnu_global'][0]:.4f} ± {d15['dnu_global'][1]:.4f} cm^-1")

    # ----------- 绘图：原始/基线/振荡/Δν(SNR筛选) -----------
    for tag, D in [("10deg", d10), ("15deg", d15)]:
        fig = plt.figure(figsize=(8,5))
        plt.plot(D['wn'], D['R'], lw=1.0, label='原始')
        plt.plot(D['wn'], D['baseline'], lw=1.0, label='基线')
        plt.xlabel('Wavenumber (cm$^{-1}$)'); plt.ylabel('Reflectance')
        plt.title(f'{tag} 原始谱与基线（ROI 内）')
        plt.legend(); plt.tight_layout()
        fig.savefig(OUTDIR/f'spectrum_baseline_{tag}_v2.png', dpi=160); plt.close(fig)

        fig = plt.figure(figsize=(8,4))
        plt.plot(D['wn'], D['R']-D['baseline'], lw=1.0)
        plt.xlabel('Wavenumber (cm$^{-1}$)'); plt.ylabel('R - baseline')
        plt.title(f'{tag} 振荡分量（ROI 内）')
        plt.tight_layout()
        fig.savefig(OUTDIR/f'spectrum_osc_{tag}_v2.png', dpi=160); plt.close(fig)

        fig = plt.figure(figsize=(8,4))
        plt.plot(D['centers'], D['dnu_local'], lw=1.0)
        plt.xlabel('Wavenumber center (cm$^{-1}$)'); plt.ylabel('Local Δν (cm$^{-1}$)')
        plt.title(f'{tag} 局部 Δν(滑窗FFT+零填充，已按 SNR≥{SNR_THRESH} 筛选)')
        plt.tight_layout()
        fig.savefig(OUTDIR/f'delta_local_{tag}_v2.png', dpi=160); plt.close(fig)

    # ----------- 方案 A：假设 n 的厚度（推荐） -----------
    th1, th2 = THETA1_DEG, THETA2_DEG
    t10 = thickness_from_dnu(d10['dnu_local'], MATERIAL_N, th1, N0_EXT)
    t15 = thickness_from_dnu(d15['dnu_local'], MATERIAL_N, th2, N0_EXT)
    t_all = np.r_[t10, t15]  # cm
    t_medA, (t_loA, t_hiA) = robust_median_ci_block(t_all, N_BOOT, BLOCK, RANDOM_SEED)
    print(f"[A·assume_n] n={MATERIAL_N:.3f} → 厚度 t ≈ {t_medA*1e4:.2f} μm (95%CI: {t_loA*1e4:.2f}~{t_hiA*1e4:.2f} μm)")

    # ===== 可靠性体检：一致性 + 稳定性 + 敏感性（简版） =====
    def _cv(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0: return np.nan
        med = np.median(arr);
        sd = np.std(arr)
        return float(sd / max(abs(med), 1e-12))

    def _slope_per_1k(x, y):
        # 对 t(ν) 做线性拟合，返回每 1000 cm^-1 的 μm 变化量
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2: return np.nan
        p = np.polyfit(x[m], y[m], 1)  # y≈p[0]*x + p[1]
        return float(p[0] * 1000.0)

    def _cos_tp(n, theta_deg, n0=1.0):
        th = math.radians(theta_deg)
        s = n0 * math.sin(th) / max(n, 1e-12)
        return np.sqrt(np.maximum(0.0, 1.0 - s * s))

    def _pred_dnu(t_med_cm, n, theta_deg, n0=1.0):
        ctp = _cos_tp(n, theta_deg, n0)
        return 1.0 / (2.0 * n * np.maximum(ctp, 1e-12) * np.maximum(t_med_cm, 1e-12))

    def _mdape(pred, obs):
        m = np.isfinite(pred) & np.isfinite(obs) & (obs > 0)
        if m.sum() == 0: return np.nan
        return float(np.median(np.abs((pred[m] - obs[m]) / obs[m])))

    # 1) Δν 平台稳定度（CV）
    cv10 = _cv(d10['dnu_local'])
    cv15 = _cv(d15['dnu_local'])

    # 2) 两角厚度一致性
    t10_um = thickness_from_dnu(d10['dnu_local'], MATERIAL_N, THETA1_DEG, N0_EXT) * 1e4
    t15_um = thickness_from_dnu(d15['dnu_local'], MATERIAL_N, THETA2_DEG, N0_EXT) * 1e4
    t10_med, t15_med = np.median(t10_um[np.isfinite(t10_um)]), np.median(t15_um[np.isfinite(t15_um)])
    t_all_med = float(np.median(np.r_[t10_um, t15_um]))
    angle_gap = abs(t10_med - t15_med) / max(t_all_med, 1e-12)

    # 3) t(ν) 平坦度（斜率）
    xsA = np.r_[d10['centers'], d15['centers']]
    tA_um = np.r_[t10_um, t15_um]
    order = np.argsort(xsA)
    slope_um_per_1k = _slope_per_1k(xsA[order], tA_um[order])

    # 4) 前向一致性（Δν 预测 vs 观测）
    t_med_cm = t_medA  # cm
    pred10 = _pred_dnu(t_med_cm, MATERIAL_N, THETA1_DEG, N0_EXT)
    pred15 = _pred_dnu(t_med_cm, MATERIAL_N, THETA2_DEG, N0_EXT)
    mdape10 = _mdape(np.full_like(d10['dnu_local'], pred10), d10['dnu_local'])
    mdape15 = _mdape(np.full_like(d15['dnu_local'], pred15), d15['dnu_local'])

    # 5) 先验 n 的敏感性（±0.1）
    def _t_med_given_n(n_try):
        a = thickness_from_dnu(d10['dnu_local'], n_try, THETA1_DEG, N0_EXT)
        b = thickness_from_dnu(d15['dnu_local'], n_try, THETA2_DEG, N0_EXT)
        return float(np.median(np.r_[a, b]))

    t_med_n_m = _t_med_given_n(MATERIAL_N - 0.10) * 1e4
    t_med_n_p = _t_med_given_n(MATERIAL_N + 0.10) * 1e4
    sens_dn_pct = 0.5 * (abs(t_med_n_m - t_all_med) + abs(t_med_n_p - t_all_med)) / max(t_all_med, 1e-12)


    # 保存 CSV 报告
    rep = pd.DataFrame([{
        'cv_dnu_10': cv10, 'cv_dnu_15': cv15,
        'angle_gap': angle_gap,
        'slope_um_per_1k': slope_um_per_1k,
        'mdape10': mdape10, 'mdape15': mdape15,
        't_med_um': t_all_med, 't_ci_lo_um': t_loA * 1e4, 't_ci_hi_um': t_hiA * 1e4,
        'sens_dn_pct_pm0.10': sens_dn_pct,
        'n_assumed': MATERIAL_N
    }])
    rep.to_csv(OUTDIR / 'reliability_summary.csv', index=False, encoding='utf-8-sig')

    # 保存 t(ν) 曲线
    fig = plt.figure(figsize=(8,4))
    xsA = np.r_[d10['centers'], d15['centers']]
    tA_um = np.r_[t10, t15]*1e4
    order = np.argsort(xsA)
    plt.plot(xsA[order], tA_um[order], lw=1.0)
    plt.xlabel('Wavenumber (cm$^{-1}$)'); plt.ylabel('t (μm)')
    plt.title(f'厚度 t(ν) – 模式A（n={MATERIAL_N:.2f}）')
    plt.tight_layout()
    plt.savefig(OUTDIR/'t_of_nu_assumeN_v2.png', dpi=160); plt.close(fig)

    # ----------- 方案 B：双角消元反 n（备选） -----------
    xsB, nB, tB = np.array([]), np.array([]), np.array([])
    if MODE == 'dual_angle':
        # 对齐两条 Δν 至公共中心点
        def _align(x1,y1,x2,y2):
            lo = max(np.min(x1), np.min(x2)); hi = min(np.max(x1), np.max(x2))
            if not (np.isfinite(lo) and np.isfinite(hi) and hi>lo):
                return np.array([]), np.array([]), np.array([])
            xs = x1 if len(x1)>=len(x2) else x2
            xs = xs[(xs>=lo)&(xs<=hi)]
            f1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate', assume_sorted=True)
            f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate', assume_sorted=True)
            return xs, f1(xs), f2(xs)
        xsB, d10B, d15B = _align(d10['centers'], d10['dnu_local'], d15['centers'], d15['dnu_local'])
        C1 = 1.0/(2.0*np.maximum(d10B,1e-12))
        C2 = 1.0/(2.0*np.maximum(d15B,1e-12))
        th1r, th2r = math.radians(THETA1_DEG), math.radians(THETA2_DEG)
        rho = C2/np.maximum(C1,1e-12)
        nB = np.array([solve_n_from_ratio(r, th1r, th2r, N0_EXT, N_SEARCH_MIN, N_SEARCH_MAX) for r in rho])
        c1 = np.array([cos_theta_t(nv, th1r, N0_EXT) for nv in nB])
        tB = C1/np.maximum(nB*np.maximum(c1,1e-12),1e-12)
        ok = np.isfinite(nB) & np.isfinite(tB) & (tB>0)
        xsB, nB, tB = xsB[ok], nB[ok], tB[ok]
        t_medB, (t_loB, t_hiB) = robust_median_ci_block(tB, N_BOOT, BLOCK, RANDOM_SEED)
        print(f"[B·dual_angle] 厚度 t ≈ {t_medB*1e4:.2f} μm (95%CI: {t_loB*1e4:.2f}~{t_hiB*1e4:.2f} μm)；n(ν) 见图")
        # 图：n 与 t
        fig = plt.figure(figsize=(8,4))
        plt.plot(xsB, nB, lw=1.0)
        plt.xlabel('Wavenumber (cm$^{-1}$)'); plt.ylabel('n')
        plt.title('折射率 n(ν) – 模式B（双角消元）')
        plt.tight_layout(); plt.savefig(OUTDIR/'n_of_nu_dual_v2.png', dpi=160); plt.close(fig)
        fig = plt.figure(figsize=(8,4))
        plt.plot(xsB, tB*1e4, lw=1.0)
        plt.xlabel('Wavenumber (cm$^{-1}$)'); plt.ylabel('t (μm)')
        plt.title('厚度 t(ν) – 模式B（双角消元）')
        plt.tight_layout(); plt.savefig(OUTDIR/'t_of_nu_dual_v2.png', dpi=160); plt.close(fig)

    # ----------- 导出 Excel -----------
    out_xlsx = OUTDIR/'result_v2.xlsx'
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
        # ROI/Δν/厚度（A）
        pd.DataFrame({
            'center_cm^-1': d10['centers'], 'delta_nu_10': d10['dnu_local'], 'snr_10': d10['snr']
        }).to_excel(w, index=False, sheet_name='local_Dnu_10deg')
        pd.DataFrame({
            'center_cm^-1': d15['centers'], 'delta_nu_15': d15['dnu_local'], 'snr_15': d15['snr']
        }).to_excel(w, index=False, sheet_name='local_Dnu_15deg')
        pd.DataFrame({
            't_um_assumeN': np.r_[t10, t15]*1e4,
            'center_cm^-1': np.r_[d10['centers'], d15['centers']]
        }).to_excel(w, index=False, sheet_name='t_of_nu_assumeN')
        # 若有 B 模式
        if MODE == 'dual_angle' and len(xsB)>0:
            pd.DataFrame({'center_cm^-1': xsB, 'n_dual': nB, 't_um_dual': tB*1e4}).to_excel(w, index=False, sheet_name='dual_angle')
        # 全局 Δν 粗估
        pd.DataFrame({
            'angle': ['10deg', '15deg'],
            'Dnu_med_cm^-1': [d10['dnu_global'][0], d15['dnu_global'][0]],
            'Dnu_sigma_cm^-1': [d10['dnu_global'][1], d15['dnu_global'][1]],
        }).to_excel(w, index=False, sheet_name='global_Dnu_peak')
    print(f"[OK] 已导出：{out_xlsx} 与图像到 {OUTDIR.resolve()}")


if __name__ == '__main__':
    solve_dual_angle_and_export(FILE_10, FILE_15, SHEET_10, SHEET_15)
