import os
import glob
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =========================
# 0) 工具：地理换算（稳健版）
# =========================

R_EARTH = 6371000.0
G = 9.80665


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def deg_per_meter_lat() -> float:
    """纬度方向：每米对应多少度（近似）"""
    return 180.0 / (math.pi * R_EARTH)


def deg_per_meter_lon(lat_deg: float) -> float:
    """
    经度方向：每米对应多少度（稳健版）
    避免 cos(lat)≈0 或 lat NaN 导致 1/cos 爆炸
    """
    lat = math.radians(float(lat_deg))
    c = math.cos(lat)

    if (not math.isfinite(c)) or (abs(c) < 1e-3):
        c = 1e-3  # 钳制最小 cos，避免爆炸
    return 180.0 / (math.pi * R_EARTH * c)


def wrap_lon_deg(lon: float) -> float:
    """把经度 wrap 到 [0,360)（OSCAR 数据是 0~360）"""
    lon = float(lon)
    if not math.isfinite(lon):
        return 0.0
    return lon % 360.0


def wrap_lon_diff(dlon: np.ndarray) -> np.ndarray:
    """把经度差 wrap 到 [-180,180]，避免跨 0/360 导致假大位移"""
    dlon = np.asarray(dlon, dtype=float)
    return (dlon + 180.0) % 360.0 - 180.0


def ll_to_xy_m(lon_deg: np.ndarray, lat_deg: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    经纬度 -> 局部平面 x/y（米），以(lon0,lat0)为原点
    注意：经度差用 wrap_lon_diff，避免 0/360 边界造成巨大假位移
    """
    lon_deg = np.asarray(lon_deg, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)

    dlon = wrap_lon_diff(lon_deg - float(lon0))
    dlat = lat_deg - float(lat0)

    x = dlon / deg_per_meter_lon(lat0)
    y = dlat / deg_per_meter_lat()
    return x, y


def pick_xr_engine() -> str:
    """优先使用 netcdf4，其次 h5netcdf/scipy；若都不可用返回空字符串交给 xarray 自己猜。"""
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            xr.backends.plugins.get_backend(eng)
            return eng
        except Exception:
            pass
    return ""


# =========================
# 1) 海水密度、阻力系数
# =========================

def rho_seawater(z_m: float) -> float:
    """
    论文密度近似（线性）：
        rho(g/cm^3) = 1.022 - 0.01 * z/1500
    z<0 表示水下；返回 kg/m^3
    """
    z_m = float(z_m)
    rho_gcm3 = 1.022 - 0.01 * (z_m / 1500.0)
    return float(rho_gcm3 * 1000.0)


def Cd_piecewise(Re: float) -> float:
    """
    分段阻力系数（钝体经验）：
    - Re < 1:        Cd = 24/Re
    - 1 <= Re < 1e3: Cd = 24/Re * (1 + 0.15 Re^0.687)
    - Re >= 1e3:     Cd ~ 0.44
    """
    Re = max(float(Re), 1e-9)
    if Re < 1.0:
        return 24.0 / Re
    if Re < 1000.0:
        return 24.0 / Re * (1.0 + 0.15 * (Re ** 0.687))
    return 0.44


def smooth_quad(v: float, eps: float = 1e-3) -> float:
    """
    光滑版 v|v|，减少速度过零时数值抖动
    v|v| -> v*sqrt(v^2 + eps^2)
    """
    v = float(v)
    return float(v * math.sqrt(v * v + eps * eps))


# =========================
# 2) 潜水器参数
# =========================

@dataclass
class SubParams:
    m_struct: float = 10500.0
    mw0: float = 2560.0
    V: float = 12.5

    drain_rate: float = 0.8

    L_char: float = 2.6
    mu: float = 1.0e-3

    A_horz: float = 6.8
    A_vert: float = 6.8

    def mw(self, t: float, can_drain: bool) -> float:
        if not can_drain:
            return self.mw0
        return max(self.mw0 - self.drain_rate * t, 0.0)

    def total_mass(self, t: float, can_drain: bool) -> float:
        return self.m_struct + self.mw(t, can_drain)

    def Re_from_speed(self, rho: float, speed: float) -> float:
        return float(rho * max(float(speed), 0.0) * self.L_char / max(self.mu, 1e-12))


# =========================
# 3) numpy 双线性插值（稳健）
# =========================

def bilinear_interp_2d(xgrid: np.ndarray, ygrid: np.ndarray, F: np.ndarray, x: float, y: float) -> float:
    """
    规则网格双线性插值
    xgrid: (nx,) 单调递增
    ygrid: (ny,) 单调递增
    F:     (ny, nx)
    """
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.asarray(ygrid, dtype=float)
    F = np.asarray(F, dtype=float)

    # 若网格异常，直接兜底
    if (xgrid.size < 2) or (ygrid.size < 2):
        return 0.0

    x = float(np.clip(x, xgrid[0], xgrid[-1]))
    y = float(np.clip(y, ygrid[0], ygrid[-1]))

    ix = int(np.searchsorted(xgrid, x, side="right") - 1)
    iy = int(np.searchsorted(ygrid, y, side="right") - 1)

    ix = max(0, min(ix, len(xgrid) - 2))
    iy = max(0, min(iy, len(ygrid) - 2))

    x0, x1 = xgrid[ix], xgrid[ix + 1]
    y0, y1 = ygrid[iy], ygrid[iy + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    f00 = F[iy, ix]
    f10 = F[iy, ix + 1]
    f01 = F[iy + 1, ix]
    f11 = F[iy + 1, ix + 1]

    val = (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11
    if not np.isfinite(val):
        return 0.0
    return float(val)


# =========================
# 4) Bathymetry（bbox子集 + numpy插值）
# =========================

@dataclass
class BathymetryFieldFast:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray  # (ny,nx) <=0

    @staticmethod
    def load_bbox(path_nc: str, lon_min: float, lon_max: float, lat_min: float, lat_max: float, engine: str) -> "BathymetryFieldFast":
        if not os.path.exists(path_nc):
            raise FileNotFoundError(f"ETOPO 文件不存在：{path_nc}")

        kw = {"decode_times": False}
        if engine:
            kw["engine"] = engine

        ds = xr.open_dataset(path_nc, **kw)
        sub = ds["z"].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)).load()
        ds.close()

        lon = sub["lon"].values.astype(float)
        lat = sub["lat"].values.astype(float)
        Z = sub.values.astype(float)

        if not np.all(np.diff(lon) > 0):
            idx = np.argsort(lon)
            lon = lon[idx]
            Z = Z[:, idx]
        if not np.all(np.diff(lat) > 0):
            idy = np.argsort(lat)
            lat = lat[idy]
            Z = Z[idy, :]

        Z = np.minimum(Z, 0.0)
        Z[~np.isfinite(Z)] = 0.0
        return BathymetryFieldFast(lon=lon, lat=lat, z=Z)

    def bottom_z(self, lon: float, lat: float) -> float:
        return bilinear_interp_2d(self.lon, self.lat, self.z, lon, lat)


def adjust_initial_depth_to_water(bathy: BathymetryFieldFast, lon0: float, lat0: float, z0: float, clearance: float = 200.0) -> float:
    zb = bathy.bottom_z(lon0, lat0)
    if z0 <= zb:
        z_new = zb + clearance
        print(f"[WARN] 初始点在海底内：z0={z0:.1f}, bottom={zb:.1f} -> 调整为 z={z_new:.1f}")
        return float(z_new)
    return float(z0)


# =========================
# 5) OSCAR（多日堆叠 + NaN修复 + 时间插值）
# =========================

@dataclass
class OscarFieldFast:
    lon: np.ndarray
    lat: np.ndarray
    u: np.ndarray  # (nt, ny, nx)
    v: np.ndarray  # (nt, ny, nx)

    @staticmethod
    def load_stack(files: List[str], lon_min: float, lon_max: float, lat_min: float, lat_max: float, engine: str) -> "OscarFieldFast":
        if len(files) == 0:
            raise FileNotFoundError("OSCAR 文件列表为空")

        def _to_0360(x: float) -> float:
            x = float(x)
            return x if x >= 0 else x + 360.0

        lon_min2 = _to_0360(lon_min)
        lon_max2 = _to_0360(lon_max)

        U_list, V_list = [], []
        lon_ref, lat_ref = None, None

        for fp in files:
            kw = {"decode_times": True}
            if engine:
                kw["engine"] = engine

            ds = xr.open_dataset(fp, **kw)

            # 让 longitude/latitude 具有“坐标索引”，避免 .sel 把 slice 当整数
            if ("longitude" in ds.dims) and ("lon" in ds.coords) and ("longitude" not in ds.indexes):
                ds = ds.set_index(longitude="lon")
            if ("latitude" in ds.dims) and ("lat" in ds.coords) and ("latitude" not in ds.indexes):
                ds = ds.set_index(latitude="lat")

            sub = ds[["u", "v"]].sel(
                longitude=slice(lon_min2, lon_max2),
                latitude=slice(lat_min, lat_max),
            )

            sub = sub.rename({"longitude": "lon", "latitude": "lat"}).transpose("time", "lat", "lon")
            sub = sub.load()

            if lon_ref is None:
                lon_ref = sub["lon"].values.astype(float)
                lat_ref = sub["lat"].values.astype(float)

            u_arr = sub["u"].values.astype(float)[0]
            v_arr = sub["v"].values.astype(float)[0]

            # 缺测/异常 -> NaN
            u_arr[~np.isfinite(u_arr)] = np.nan
            v_arr[~np.isfinite(v_arr)] = np.nan
            # 海流强度不可能 > 10 m/s（极端防御）
            u_arr[np.abs(u_arr) > 10.0] = np.nan
            v_arr[np.abs(v_arr) > 10.0] = np.nan

            U_list.append(u_arr)
            V_list.append(v_arr)

            ds.close()

        U = np.stack(U_list, axis=0)
        V = np.stack(V_list, axis=0)

        lon = lon_ref
        lat = lat_ref

        if not np.all(np.diff(lon) > 0):
            idx = np.argsort(lon)
            lon = lon[idx]
            U = U[:, :, idx]
            V = V[:, :, idx]
        if not np.all(np.diff(lat) > 0):
            idy = np.argsort(lat)
            lat = lat[idy]
            U = U[:, idy, :]
            V = V[:, idy, :]

        return OscarFieldFast(lon=lon, lat=lat, u=U, v=V)

    @staticmethod
    def _to_0360_lon(lon_deg: float) -> float:
        lon_deg = float(lon_deg)
        return lon_deg if lon_deg >= 0 else lon_deg + 360.0

    def get_uv_day(self, lon_deg: float, lat_deg: float, day_idx: int) -> Tuple[float, float]:
        nt = self.u.shape[0]
        day_idx = int(np.clip(day_idx, 0, nt - 1))
        lon0360 = self._to_0360_lon(lon_deg)

        uc = bilinear_interp_2d(self.lon, self.lat, self.u[day_idx], lon0360, lat_deg)
        vc = bilinear_interp_2d(self.lon, self.lat, self.v[day_idx], lon0360, lat_deg)
        return float(uc), float(vc)

    def get_uv_time(self, lon_deg: float, lat_deg: float, t_sec: float) -> Tuple[float, float]:
        """跨天线性插值，减少“横跳”"""
        day0 = int(max(float(t_sec), 0.0) // 86400.0)
        alpha = (max(float(t_sec), 0.0) - day0 * 86400.0) / 86400.0

        u0, v0 = self.get_uv_day(lon_deg, lat_deg, day0)
        u1, v1 = self.get_uv_day(lon_deg, lat_deg, day0 + 1)

        uc = (1.0 - alpha) * u0 + alpha * u1
        vc = (1.0 - alpha) * v0 + alpha * v1
        return float(uc), float(vc)

    def daily_region_mean(self) -> Tuple[np.ndarray, np.ndarray]:
        u_mean = np.nanmean(self.u, axis=(1, 2))
        v_mean = np.nanmean(self.v, axis=(1, 2))
        return u_mean, v_mean


def compute_daily_mean_cov(oscar: OscarFieldFast) -> Tuple[np.ndarray, np.ndarray]:
    u_mean, v_mean = oscar.daily_region_mean()

    mask = np.isfinite(u_mean) & np.isfinite(v_mean)
    X = np.vstack([u_mean[mask], v_mean[mask]]).T

    if X.shape[0] == 0:
        mu = np.zeros(2, dtype=float)
        Sigma = np.eye(2, dtype=float) * 1e-6
        return mu, Sigma

    mu = X.mean(axis=0).astype(float)

    if X.shape[0] == 1:
        Sigma = np.eye(2, dtype=float) * 1e-4
        return mu, Sigma

    Sigma = np.cov(X.T, ddof=1).astype(float)
    if not np.all(np.isfinite(Sigma)):
        Sigma = np.eye(2, dtype=float) * 1e-4
    return mu, Sigma


# =========================
# 6) 初始随机性（位置/状态）
# =========================

@dataclass
class InitialUncertainty:
    sigma_xy_m: float = 1500.0
    sigma_z_m: float = 80.0
    sigma_u_mps: float = 0.25
    sigma_w_mps: float = 0.08
    corr_uv: float = 0.2

    use_uniform_disk: bool = False
    disk_radius_m: float = 2500.0


def sample_xy_offset_m(rng: np.random.Generator, cfg: InitialUncertainty) -> Tuple[float, float]:
    if cfg.use_uniform_disk:
        u = rng.random()
        r = cfg.disk_radius_m * math.sqrt(u)
        theta = 2.0 * math.pi * rng.random()
        return float(r * math.cos(theta)), float(r * math.sin(theta))

    return float(rng.normal(0.0, cfg.sigma_xy_m)), float(rng.normal(0.0, cfg.sigma_xy_m))


def sample_initial_state(
    rng: np.random.Generator,
    s0_mean: np.ndarray,
    cfg: InitialUncertainty
) -> Tuple[np.ndarray, Dict[str, float]]:
    lon0, lat0, z0, u0, v0, w0 = map(float, s0_mean)

    dx_m, dy_m = sample_xy_offset_m(rng, cfg)
    dlon = dx_m * deg_per_meter_lon(lat0)
    dlat = dy_m * deg_per_meter_lat()

    lon = wrap_lon_deg(lon0 + dlon)
    lat = float(np.clip(lat0 + dlat, -89.9, 89.9))

    dz = float(rng.normal(0.0, cfg.sigma_z_m))
    z = z0 + dz

    if abs(cfg.corr_uv) < 1e-12:
        du = float(rng.normal(0.0, cfg.sigma_u_mps))
        dv = float(rng.normal(0.0, cfg.sigma_u_mps))
    else:
        s2 = cfg.sigma_u_mps ** 2
        cov = np.array([[s2, cfg.corr_uv * s2],
                        [cfg.corr_uv * s2, s2]], dtype=float)
        du, dv = rng.multivariate_normal(mean=np.zeros(2), cov=cov)
        du, dv = float(du), float(dv)

    dw = float(rng.normal(0.0, cfg.sigma_w_mps))

    u = u0 + du
    v = v0 + dv
    w = w0 + dw

    s = np.array([lon, lat, z, u, v, w], dtype=float)

    meta = {
        "dx_m": float(dx_m), "dy_m": float(dy_m), "dz_m": float(dz),
        "du_mps": float(du), "dv_mps": float(dv), "dw_mps": float(dw),
        "lon_init": float(lon), "lat_init": float(lat), "z_init": float(z),
        "u_init": float(u), "v_init": float(v), "w_init": float(w),
    }
    return s, meta


def _make_psd_cov(S: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)

    if not np.all(np.isfinite(S)):
        return np.eye(2, dtype=float) * 1e-4

    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 0.0)
    S_psd = (V * w) @ V.T
    S_psd[0, 0] += jitter
    S_psd[1, 1] += jitter
    return S_psd


def sample_uv_bias(mu: np.ndarray, Sigma: np.ndarray, n: int, seed: int = 42, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S = _make_psd_cov(Sigma, jitter=1e-8) * float(scale) ** 2

    try:
        return rng.multivariate_normal(mean=np.zeros(2), cov=S, size=n)
    except Exception:
        diag = np.diag(S)
        diag = np.where(np.isfinite(diag) & (diag > 0), diag, 1e-4)
        std = np.sqrt(diag)
        return rng.normal(loc=0.0, scale=std, size=(n, 2))


# =========================
# 7) 动力学 + RK4（稳健）
# =========================

def rhs_location(
    t: float,
    s: np.ndarray,
    bathy: BathymetryFieldFast,
    oscar: OscarFieldFast,
    p: SubParams,
    can_drain: bool,
    uv_bias: Tuple[float, float],
) -> np.ndarray:
    lon, lat, z, u_rel, v_rel, w = map(float, s)

    # 基本检查，避免 NaN 扩散
    if not all(map(math.isfinite, [lon, lat, z, u_rel, v_rel, w])):
        return np.zeros(6, dtype=float)

    lat = float(np.clip(lat, -89.9, 89.9))
    lon = wrap_lon_deg(lon)

    uc, vc = oscar.get_uv_time(lon, lat, t)
    uc += float(uv_bias[0])
    vc += float(uv_bias[1])

    rho = rho_seawater(z)
    M = p.total_mass(t, can_drain=can_drain)

    speed_h = math.hypot(u_rel, v_rel)
    Re_h = p.Re_from_speed(rho, speed_h)
    Re_v = p.Re_from_speed(rho, abs(w))

    Cd_h = Cd_piecewise(Re_h)
    Cd_v = Cd_piecewise(Re_v)

    drag_z = 0.5 * Cd_v * rho * p.A_vert * smooth_quad(w, 1e-3)
    w_dot = (-M * G + rho * G * p.V - drag_z) / M

    u_dot = -(0.5 * Cd_h * rho * p.A_horz * smooth_quad(u_rel, 1e-3)) / M
    v_dot = -(0.5 * Cd_h * rho * p.A_horz * smooth_quad(v_rel, 1e-3)) / M

    lon_dot = (u_rel + uc) * deg_per_meter_lon(lat)
    lat_dot = (v_rel + vc) * deg_per_meter_lat()
    z_dot = w

    out = np.array([lon_dot, lat_dot, z_dot, u_dot, v_dot, w_dot], dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def rk4_step(f, t: float, s: np.ndarray, h: float, *args) -> np.ndarray:
    k1 = f(t, s, *args)
    k2 = f(t + 0.5 * h, s + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, s + 0.5 * h * k2, *args)
    k4 = f(t + h, s + h * k3, *args)
    s2 = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s2


def simulate_one_trajectory(
    s0: np.ndarray,
    bathy: BathymetryFieldFast,
    oscar: OscarFieldFast,
    p: SubParams,
    T: float,
    h: float,
    can_drain: bool,
    uv_bias: Tuple[float, float] = (0.0, 0.0),
    stop_at_surface: bool = True,
    bottom_margin: float = 1.0,
    stick_to_bottom: bool = True,
    progress_every: int = 500,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    N = int(T // h) + 1
    times = np.arange(N, dtype=float) * h
    traj = np.zeros((N, 6), dtype=float)
    traj[0] = s0

    info: Dict[str, object] = {"stop_t": np.nan, "stop_reason": ""}

    # 初始稳健化
    traj[0, 0] = wrap_lon_deg(traj[0, 0])
    traj[0, 1] = float(np.clip(traj[0, 1], -89.9, 89.9))
    if not np.all(np.isfinite(traj[0])):
        traj[0] = np.array([wrap_lon_deg(0.0), 0.0, -1000.0, 0.0, 0.0, 0.0], dtype=float)

    for i in range(1, N):
        if (i % progress_every) == 0:
            print(f"[SIM] step {i}/{N}  t={times[i]/3600:.2f} h  z={traj[i-1,2]:.1f} m")

        lon, lat, z, u_rel, v_rel, w = traj[i - 1]
        lon = wrap_lon_deg(lon)
        lat = float(np.clip(lat, -89.9, 89.9))

        zb = bathy.bottom_z(float(lon), float(lat))

        # 触底处理：钳制 z/w，但仍允许洋流推水平漂移
        if z <= (zb + bottom_margin):
            if stick_to_bottom:
                uc, vc = oscar.get_uv_time(float(lon), float(lat), float(times[i - 1]))
                uc += float(uv_bias[0])
                vc += float(uv_bias[1])

                lon_new = wrap_lon_deg(lon + uc * h * deg_per_meter_lon(lat))
                lat_new = float(np.clip(lat + vc * h * deg_per_meter_lat(), -89.9, 89.9))
                z_new = float(zb + bottom_margin)

                traj[i] = np.array([lon_new, lat_new, z_new, 0.0, 0.0, 0.0], dtype=float)
                continue

            traj[i:] = traj[i - 1]
            info["stop_t"] = float(times[i])
            info["stop_reason"] = "hit_bottom_freeze"
            break

        if stop_at_surface and z >= 0.0:
            traj[i:] = traj[i - 1]
            info["stop_t"] = float(times[i])
            info["stop_reason"] = "reach_surface_freeze"
            break

        t = float(times[i - 1])
        traj[i] = rk4_step(rhs_location, t, traj[i - 1], h, bathy, oscar, p, can_drain, uv_bias)

        # ====== 硬防御：每步后检查/钳制，避免经纬度爆炸 ======
        if not np.all(np.isfinite(traj[i])):
            traj[i:] = traj[i - 1]
            info["stop_t"] = float(times[i])
            info["stop_reason"] = "nan_or_inf"
            break

        traj[i, 1] = float(np.clip(traj[i, 1], -89.9, 89.9))
        traj[i, 0] = wrap_lon_deg(traj[i, 0])

        # 额外：若一步跳得过大，直接冻结（避免个别异常把 MC 拉爆）
        if abs(traj[i, 0] - traj[i - 1, 0]) > 5.0 and abs(wrap_lon_diff(np.array([traj[i, 0] - traj[i - 1, 0]]))[0]) > 5.0:
            traj[i:] = traj[i - 1]
            info["stop_t"] = float(times[i])
            info["stop_reason"] = "lon_jump_freeze"
            break
        if abs(traj[i, 1] - traj[i - 1, 1]) > 5.0:
            traj[i:] = traj[i - 1]
            info["stop_t"] = float(times[i])
            info["stop_reason"] = "lat_jump_freeze"
            break

    return times, traj, info


# =========================
# 8) 蒙特卡洛（稳健）
# =========================

def monte_carlo_endpoints_with_initial_randomness(
    s0_mean: np.ndarray,
    bathy: BathymetryFieldFast,
    oscar: OscarFieldFast,
    p: SubParams,
    n_mc: int,
    T: float,
    h: float,
    can_drain: bool,
    init_cfg,
    mu_flow: np.ndarray,
    Sigma_flow: np.ndarray,
    flow_scale: float = 1.0,
    seed_init: int = 2025,
    seed_flow: int = 42,
) -> pd.DataFrame:
    rng_init = np.random.default_rng(seed_init)
    biases = sample_uv_bias(mu_flow, Sigma_flow, n=n_mc, seed=seed_flow, scale=flow_scale)

    rows = []
    for k in range(n_mc):
        s0_k, meta = sample_initial_state(rng_init, s0_mean, init_cfg)
        du, dv = float(biases[k, 0]), float(biases[k, 1])

        times, traj, info = simulate_one_trajectory(
            s0=s0_k,
            bathy=bathy,
            oscar=oscar,
            p=p,
            T=T,
            h=h,
            can_drain=can_drain,
            uv_bias=(du, dv),
            stop_at_surface=True,
            bottom_margin=1.0,
            stick_to_bottom=True,
            progress_every=max(5000, int((T // h) // 10)),
        )

        lon_end = float(traj[-1, 0])
        lat_end = float(traj[-1, 1])
        z_end = float(traj[-1, 2])

        # 兜底：若 endpoint 非有限，直接标记 NaN
        if not (math.isfinite(lon_end) and math.isfinite(lat_end) and math.isfinite(z_end)):
            lon_end, lat_end, z_end = np.nan, np.nan, np.nan

        rows.append({
            "mc": k,
            "du_bias": du,
            "dv_bias": dv,
            "lon_end": lon_end,
            "lat_end": lat_end,
            "z_end": z_end,
            "stop_t": float(info["stop_t"]) if math.isfinite(float(info["stop_t"])) else np.nan,
            "stop_reason": str(info.get("stop_reason", "")),
            **meta
        })

        if (k + 1) % max(1, n_mc // 10) == 0:
            print(f"[MC] {k+1}/{n_mc} done")

    df = pd.DataFrame(rows)

    # 画图/统计前：过滤极端值（避免 1e60 拉爆）
    mask = np.isfinite(df["lon_end"]) & np.isfinite(df["lat_end"])
    df = df.loc[mask].copy()
    return df


# =========================
# 9) 画图（稳健：过滤异常点）
# =========================

def plot_topdown_trajectory_m(df_traj: pd.DataFrame, lon0: float, lat0: float, out_png: str) -> None:
    lon = df_traj["lon"].to_numpy(float)
    lat = df_traj["lat"].to_numpy(float)
    x, y = ll_to_xy_m(lon, lat, lon0, lat0)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(6.2, 5.2))
    plt.plot(x, y, linewidth=2.2)
    plt.scatter([0], [0], s=80, facecolors="none", edgecolors="magenta", linewidths=2)
    plt.xlabel("x East (m)")
    plt.ylabel("y North (m)")
    plt.title("Trajectory (Top-down, meters)")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_endpoints_scatter_and_heat(df_mc: pd.DataFrame, lon0: float, lat0: float, out_png: str) -> None:
    lon = df_mc["lon_end"].to_numpy(float)
    lat = df_mc["lat_end"].to_numpy(float)
    x, y = ll_to_xy_m(lon, lat, lon0, lat0)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(x, y, s=10, alpha=0.5)
    plt.scatter([0], [0], s=80, facecolors="none", edgecolors="magenta", linewidths=2)
    plt.xlabel("x East (m)")
    plt.ylabel("y North (m)")
    plt.title("MC endpoints scatter (meters)")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.2, 5.2))
    bins = 60
    if len(x) == 0:
        H = np.zeros((bins, bins), dtype=float)
        xedges = np.linspace(-1, 1, bins + 1)
        yedges = np.linspace(-1, 1, bins + 1)
    else:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(H.T, origin="lower", extent=extent, aspect="equal")
    plt.scatter([0], [0], s=80, facecolors="none", edgecolors="magenta", linewidths=2)
    plt.xlabel("x East (m)")
    plt.ylabel("y North (m)")
    plt.title("MC endpoints probability heatmap")
    plt.colorbar(label="counts")
    out_png2 = out_png.replace(".png", "_heat.png")
    plt.savefig(out_png2, dpi=220, bbox_inches="tight")
    plt.close()


def plot_schematic_bathy_current(
    bathy: BathymetryFieldFast,
    oscar: OscarFieldFast,
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    t_sec: float,
    out_png: str,
    quiver_step: int = 3
) -> None:
    LonB, LatB = np.meshgrid(bathy.lon, bathy.lat)
    Z = bathy.z

    day_idx = int(max(float(t_sec), 0.0) // 86400.0)
    day_idx = max(0, min(day_idx, oscar.u.shape[0] - 1))

    lonO = oscar.lon
    latO = oscar.lat
    U = oscar.u[day_idx]
    V = oscar.v[day_idx]

    lonO2 = lonO[::quiver_step]
    latO2 = latO[::quiver_step]
    U2 = U[::quiver_step, ::quiver_step]
    V2 = V[::quiver_step, ::quiver_step]
    LonO2, LatO2 = np.meshgrid(lonO2, latO2)

    plt.figure(figsize=(7.2, 5.8))
    plt.pcolormesh(LonB, LatB, Z, shading="auto")
    plt.colorbar(label="ETOPO z (m, <=0)")

    levels = [-4000, -3000, -2000, -1000, -500, -200]
    cs = plt.contour(LonB, LatB, Z, levels=levels, linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%d")

    plt.quiver(LonO2, LatO2, U2, V2, scale=10.0, width=0.0025)

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Schematic: Bathymetry + OSCAR currents (lon-lat)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# 10) main
# =========================

def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT, "data")
    OUT_DIR = os.path.join(ROOT, "output")
    ensure_dir(OUT_DIR)

    XR_ENGINE = pick_xr_engine()
    print("[INFO] xarray engine =", XR_ENGINE if XR_ENGINE else "(auto-guess)")

    ETOPO_PATH = os.path.join(DATA_DIR, "etopo", "ETOPO_2022_v1_60s_N90W180_bed.nc")
    OSCAR_GLOB = os.path.join(DATA_DIR, "oscar", "oscar_currents_interim_*.nc")

    # Ionian Sea bbox
    LON_MIN, LON_MAX = 14.0, 22.5
    LAT_MIN, LAT_MAX = 33.0, 40.5
    PAD = 1.0

    # 模拟时长/步长
    T = 3 * 3600.0
    h = 1
    CAN_DRAIN = False

    # 初始状态（可改）
    lon0, lat0 = 19, 37.35
    z0 = -2000.0
    u0, v0 = 0.8, 0.2
    w0 = 0.0

    p = SubParams(
        m_struct=10500.0,
        mw0=2560.0,
        V=12.55,
        drain_rate=0.8,
        L_char=2.6,
        mu=1e-3,
        A_horz=6.8,
        A_vert=6.8
    )

    print("[INFO] loading ETOPO (bbox subset) ...")
    bathy = BathymetryFieldFast.load_bbox(
        ETOPO_PATH,
        lon_min=LON_MIN - PAD, lon_max=LON_MAX + PAD,
        lat_min=LAT_MIN - PAD, lat_max=LAT_MAX + PAD,
        engine=XR_ENGINE
    )
    z0 = adjust_initial_depth_to_water(bathy, lon0, lat0, z0, clearance=200.0)

    oscar_files = sorted([fp for fp in glob.glob(OSCAR_GLOB) if fp.lower().endswith(".nc")])
    print(f"[INFO] found {len(oscar_files)} OSCAR files")
    if len(oscar_files) == 0:
        raise FileNotFoundError(f"未找到 OSCAR 文件：{OSCAR_GLOB}")

    print("[INFO] loading OSCAR stack (bbox subset) ...")
    oscar = OscarFieldFast.load_stack(oscar_files, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, engine=XR_ENGINE)

    mu_flow, Sigma_flow = compute_daily_mean_cov(oscar)
    print("[INFO] OSCAR daily mean (u,v) =", mu_flow)
    print("[INFO] OSCAR daily cov Sigma =\n", Sigma_flow)

    out_schem = os.path.join(OUT_DIR, "schematic_bathy_currents.png")
    plot_schematic_bathy_current(
        bathy=bathy,
        oscar=oscar,
        lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX,
        t_sec=0.0,
        out_png=out_schem,
        quiver_step=3
    )
    print("[INFO] saved:", out_schem)

    s0 = np.array([wrap_lon_deg(lon0), float(np.clip(lat0, -89.9, 89.9)), float(z0), float(u0), float(v0), float(w0)], dtype=float)

    print("[INFO] simulate one trajectory ...")
    times, traj, info = simulate_one_trajectory(
        s0=s0,
        bathy=bathy,
        oscar=oscar,
        p=p,
        T=T,
        h=h,
        can_drain=CAN_DRAIN,
        uv_bias=(0.0, 0.0),
        stop_at_surface=True,
        bottom_margin=1.0,
        stick_to_bottom=True,
        progress_every=500
    )

    df_traj = pd.DataFrame({
        "time": times,
        "lon": traj[:, 0],
        "lat": traj[:, 1],
        "z": traj[:, 2],
        "u_rel": traj[:, 3],
        "v_rel": traj[:, 4],
        "w": traj[:, 5],
    })
    out_csv = os.path.join(OUT_DIR, "one_trajectory.csv")
    df_traj.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[INFO] saved:", out_csv)
    if info.get("stop_reason", ""):
        print("[INFO] stop:", info)

    out_png = os.path.join(OUT_DIR, "traj_topdown_meters.png")
    plot_topdown_trajectory_m(df_traj, s0[0], s0[1], out_png)
    print("[INFO] saved:", out_png)

    init_cfg = InitialUncertainty(
        sigma_xy_m=1500.0,
        sigma_z_m=80.0,
        sigma_u_mps=0.25,
        sigma_w_mps=0.08,
        corr_uv=0.2,
        use_uniform_disk=False,
        disk_radius_m=2500.0
    )

    N_MC = 50
    FLOW_SCALE = 1.0

    print(f"[INFO] Monte Carlo with initial randomness: N={N_MC} ...")
    df_mc = monte_carlo_endpoints_with_initial_randomness(
        s0_mean=s0,
        bathy=bathy,
        oscar=oscar,
        p=p,
        n_mc=N_MC,
        T=T,
        h=h,
        can_drain=CAN_DRAIN,
        init_cfg=init_cfg,
        mu_flow=mu_flow,
        Sigma_flow=Sigma_flow,
        flow_scale=FLOW_SCALE,
        seed_init=2025,
        seed_flow=42
    )

    # Debug：检查是否还有极端点
    tmp = df_mc[["mc", "lon_end", "lat_end", "z_end"]].copy()
    bad = tmp[~np.isfinite(tmp["lon_end"]) | ~np.isfinite(tmp["lat_end"]) |
              (np.abs(tmp["lon_end"]) > 1e6) | (np.abs(tmp["lat_end"]) > 1e6)]
    print("[DEBUG] bad endpoints count =", len(bad))
    if len(bad) > 0:
        print(bad.head(20))

    out_mc = os.path.join(OUT_DIR, "mc_endpoints_with_init_randomness.csv")
    df_mc.to_csv(out_mc, index=False, encoding="utf-8-sig")
    print("[INFO] saved:", out_mc)

    out_mc_png = os.path.join(OUT_DIR, "mc_endpoints_scatter.png")
    plot_endpoints_scatter_and_heat(df_mc, s0[0], s0[1], out_mc_png)
    print("[INFO] saved:", out_mc_png, "and heatmap png")

    lon_end = float(df_traj["lon"].iloc[-1])
    lat_end = float(df_traj["lat"].iloc[-1])
    dx_m = float(wrap_lon_diff(np.array([lon_end - s0[0]]))[0]) / deg_per_meter_lon((s0[1] + lat_end) / 2.0)
    dy_m = float(lat_end - s0[1]) / deg_per_meter_lat()
    dist = math.hypot(dx_m, dy_m)
    print(f"[INFO] one trajectory displacement: dx={dx_m:.1f} m, dy={dy_m:.1f} m, |d|={dist/1000:.2f} km")
    print("[DONE]")


if __name__ == "__main__":
    main()
