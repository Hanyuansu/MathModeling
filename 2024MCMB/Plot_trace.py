# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


# =========================
# 1) 经纬度 -> 局部米坐标（尺度匹配关键）
# =========================
R_EARTH = 6371000.0

def deg_per_meter_lat():
    return 180.0 / (math.pi * R_EARTH)

def deg_per_meter_lon(lat_deg: float):
    c = math.cos(math.radians(float(lat_deg)))
    if (not math.isfinite(c)) or abs(c) < 1e-3:
        c = 1e-3
    return 180.0 / (math.pi * R_EARTH * c)

def wrap_lon_diff(dlon_deg):
    dlon_deg = np.asarray(dlon_deg, dtype=float)
    return (dlon_deg + 180.0) % 360.0 - 180.0

def ll_to_xy_m(lon_deg, lat_deg, lon0, lat0):
    lon_deg = np.asarray(lon_deg, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)
    dlon = wrap_lon_diff(lon_deg - float(lon0))
    dlat = lat_deg - float(lat0)
    x = dlon / deg_per_meter_lon(lat0)     # East (m)
    y = dlat / deg_per_meter_lat()         # North (m)
    return x, y


# =========================
# 2) 读取 ETOPO 子集（只取轨迹附近区域）
# =========================
def load_etopo_subset(etopo_path, lon_min, lon_max, lat_min, lat_max, engine="netcdf4"):
    if not os.path.exists(etopo_path):
        raise FileNotFoundError(f"ETOPO 文件不存在：{etopo_path}")

    ds = xr.open_dataset(etopo_path, decode_times=False, engine=engine)

    # ETOPO2022 60s: coords lon/lat, var z
    sub = ds["z"].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)).load()
    ds.close()

    lon = sub["lon"].values.astype(float)
    lat = sub["lat"].values.astype(float)
    Z = sub.values.astype(float)

    # 排序保证单调递增
    if not np.all(np.diff(lon) > 0):
        idx = np.argsort(lon)
        lon = lon[idx]
        Z = Z[:, idx]
    if not np.all(np.diff(lat) > 0):
        idy = np.argsort(lat)
        lat = lat[idy]
        Z = Z[idy, :]

    Z[~np.isfinite(Z)] = np.nan
    return lon, lat, Z


# =========================
# 2.1) 根据轨迹自动生成“合适大小”的裁剪框（米坐标）
# =========================
def auto_bbox_from_traj_xy(traj_df, lon0, lat0,
                           pad_ratio=2.5,
                           pad_min_m=2500.0,
                           pad_max_m=20000.0):
    """
    核心思想：先把轨迹投到局部米坐标，得到轨迹本身的 bounding box；
    再按比例 + 最小/最大限制扩边，得到“合适大小”的地形裁剪框。

    pad_ratio: 轨迹尺寸的扩边比例（建议 2~4）
    pad_min_m: 最小扩边（轨迹很短时仍保留地形上下文）
    pad_max_m: 防止轨迹很长导致裁剪区过大
    """
    lon_t = traj_df["lon"].to_numpy(float)
    lat_t = traj_df["lat"].to_numpy(float)
    xt, yt = ll_to_xy_m(lon_t, lat_t, lon0, lat0)

    xmin, xmax = float(np.min(xt)), float(np.max(xt))
    ymin, ymax = float(np.min(yt)), float(np.max(yt))

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)

    # 根据轨迹尺度自动取扩边：按最大边长
    base = max(dx, dy)
    pad = base * float(pad_ratio)
    pad = max(pad, float(pad_min_m))
    pad = min(pad, float(pad_max_m))

    # 让裁剪框尽量“方一点”更好看：用最大边长统一扩展
    half_w = dx * 0.5 + pad
    half_h = dy * 0.5 + pad

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    bbox_xy = {
        "x_min": cx - half_w,
        "x_max": cx + half_w,
        "y_min": cy - half_h,
        "y_max": cy + half_h,
        "traj_x_min": xmin, "traj_x_max": xmax,
        "traj_y_min": ymin, "traj_y_max": ymax,
        "pad_m": pad
    }
    return bbox_xy


# =========================
# 2.2) 米裁剪框 -> lon/lat 裁剪范围
# =========================
def bbox_xy_to_ll(bbox_xy, lon0, lat0):
    """
    由于 ll_to_xy_m 用的是固定 lat0 的线性近似，所以反算也很直接：
      lon = lon0 + x * deg_per_meter_lon(lat0)
      lat = lat0 + y * deg_per_meter_lat()
    """
    lon_min = float(lon0) + float(bbox_xy["x_min"]) * deg_per_meter_lon(lat0)
    lon_max = float(lon0) + float(bbox_xy["x_max"]) * deg_per_meter_lon(lat0)
    lat_min = float(lat0) + float(bbox_xy["y_min"]) * deg_per_meter_lat()
    lat_max = float(lat0) + float(bbox_xy["y_max"]) * deg_per_meter_lat()

    # 保证 min<max
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    return lon_min, lon_max, lat_min, lat_max


# =========================
# 2.3) 地形网格重采样到 200m 间距（可选）
# =========================
def resample_bathy_to_meter_grid(
    lon_grid, lat_grid, Z,
    lon0, lat0,
    dx=200.0, dy=200.0,
    method="linear",
    max_cells=2_000_000
):
    """
    将规则 lon/lat 网格重采样到局部米坐标等间距网格。
    max_cells: 防止区域太大时网格爆炸，超过阈值就自动加粗 dx/dy。
    """
    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)
    Z = np.asarray(Z, dtype=float)

    x_old = wrap_lon_diff(lon_grid - float(lon0)) / deg_per_meter_lon(lat0)
    y_old = (lat_grid - float(lat0)) / deg_per_meter_lat()

    # 单调性兜底
    if not np.all(np.diff(x_old) > 0):
        idx = np.argsort(x_old)
        x_old = x_old[idx]
        lon_grid = lon_grid[idx]
        Z = Z[:, idx]
    if not np.all(np.diff(y_old) > 0):
        idy = np.argsort(y_old)
        y_old = y_old[idy]
        lat_grid = lat_grid[idy]
        Z = Z[idy, :]

    x_min, x_max = float(np.min(x_old)), float(np.max(x_old))
    y_min, y_max = float(np.min(y_old)), float(np.max(y_old))

    # 先用目标 dx/dy 估算网格规模，太大就自动加粗
    nx = int(math.floor((x_max - x_min) / dx)) + 1
    ny = int(math.floor((y_max - y_min) / dy)) + 1
    cells = nx * ny
    if cells > int(max_cells):
        scale = math.sqrt(cells / float(max_cells))
        dx2 = dx * scale
        dy2 = dy * scale
        print(f"[WARN] resample grid too large ({cells} cells). Auto coarsen: dx {dx}->{dx2:.1f}, dy {dy}->{dy2:.1f}")
        dx, dy = dx2, dy2

    x_new = np.arange(x_min, x_max + dx, dx, dtype=float)
    y_new = np.arange(y_min, y_max + dy, dy, dtype=float)

    lon_new = float(lon0) + x_new * deg_per_meter_lon(lat0)
    lat_new = float(lat0) + y_new * deg_per_meter_lat()

    da = xr.DataArray(
        Z,
        coords={"lat": lat_grid, "lon": lon_grid},
        dims=("lat", "lon")
    )

    da_new = da.interp(
        lon=xr.DataArray(lon_new, dims="lon"),
        lat=xr.DataArray(lat_new, dims="lat"),
        method=method
    )

    Z_new = da_new.values.astype(float)
    Z_new[~np.isfinite(Z_new)] = np.nan

    print(f"[INFO] resample: new grid = ({len(lat_new)} x {len(lon_new)}), spacing=({dx:.1f}m,{dy:.1f}m)")
    return lon_new, lat_new, Z_new


# =========================
# 2.4) 保存裁剪后的地形（可选）
# =========================
def save_bathy_subset_netcdf(out_nc, lon_grid, lat_grid, Z):
    ds = xr.Dataset(
        data_vars={"z": (("lat", "lon"), Z)},
        coords={"lon": lon_grid, "lat": lat_grid},
    )
    ds.to_netcdf(out_nc)
    ds.close()
    print("[OK] saved bathy subset:", out_nc)


# =========================
# 3) 3D 绘图（地形光照 + 轨迹 + 垂直夸张）
# =========================
def plot_3d_bathy_and_traj(
    lon_grid, lat_grid, Z,
    traj_df,
    lon0, lat0,
    out_png,
    bbox_xy=None,      # 新增：用于锁定 x/y 轴范围
    downsample=2,
    z_exag=60.0,
    elev=22, azim=-55
):
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)

    Zb = Z.copy()
    Zb_sea = Zb.copy()
    Zb_sea[Zb_sea > 0] = np.nan

    if np.all(~np.isfinite(Zb_sea)):
        Zplot = Zb
    else:
        Zplot = Zb_sea

    z_valid = Zplot[np.isfinite(Zplot)]
    if z_valid.size > 0:
        print(f"[INFO] bathy stats (m): min={np.min(z_valid):.1f}, max={np.max(z_valid):.1f}, std={np.std(z_valid):.1f}")

    Lon2 = Lon[::downsample, ::downsample]
    Lat2 = Lat[::downsample, ::downsample]
    Z2   = Zplot[::downsample, ::downsample]

    X2, Y2 = ll_to_xy_m(Lon2, Lat2, lon0, lat0)

    lon_t = traj_df["lon"].to_numpy(float)
    lat_t = traj_df["lat"].to_numpy(float)
    z_t   = traj_df["z"].to_numpy(float)
    xt, yt = ll_to_xy_m(lon_t, lat_t, lon0, lat0)

    ls = LightSource(azdeg=315, altdeg=45)
    Z_for_shade = np.nan_to_num(Z2, nan=np.nanmin(z_valid) if z_valid.size else -3000.0)
    facecolors = ls.shade(Z_for_shade, cmap=plt.cm.terrain, vert_exag=1.0, blend_mode='soft')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X2, Y2, Z2,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.95
    )

    ax.plot(xt, yt, z_t, linewidth=2.5, label="trajectory")
    ax.scatter([xt[0]], [yt[0]], [z_t[0]], s=90, marker="*", label="start")
    ax.scatter([xt[-1]], [yt[-1]], [z_t[-1]], s=70, marker="o", label="end")

    ax.set_xlabel("x East (m)")
    ax.set_ylabel("y North (m)")
    ax.set_zlabel("depth / elevation (m)")
    ax.set_title("Trajectory over Cropped Bathymetry (auto-fit)")

    # 关键：锁定视野到“裁剪框”，保证轨迹占主要画面
    if bbox_xy is not None:
        ax.set_xlim(float(bbox_xy["x_min"]), float(bbox_xy["x_max"]))
        ax.set_ylim(float(bbox_xy["y_min"]), float(bbox_xy["y_max"]))

    # 垂直夸张（只改变显示比例）
    try:
        rx = np.nanmax(X2) - np.nanmin(X2)
        ry = np.nanmax(Y2) - np.nanmin(Y2)
        rz = np.nanmax(Z2) - np.nanmin(Z2)
        rz = max(float(rz), 1.0)
        ax.set_box_aspect((rx, ry, rz * float(z_exag)))
        print(f"[INFO] vertical exaggeration z_exag={z_exag}")
    except Exception as e:
        print("[WARN] set_box_aspect failed:", e)

    ax.view_init(elev=elev, azim=azim)

    z_all = np.concatenate([z_valid, z_t[np.isfinite(z_t)]]) if z_valid.size else z_t[np.isfinite(z_t)]
    if z_all.size:
        zmin = float(np.min(z_all))
        zmax = float(np.max(z_all))
        ax.set_zlim(zmin - 50, zmax + 50)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
def check_bottom_collision(lon_grid, lat_grid, Z, traj_df, safety_margin=0.0):
    """
    判定是否触底：
    clearance = z_traj - z_bathy
    - 若 clearance <= safety_margin，则认为“触底/过近”
    说明：
    - z_bathy（ETOPO）海底一般为负值
    - z_traj 若为负值表示在海平面下
    """
    da = xr.DataArray(
        Z,
        coords={"lat": lat_grid, "lon": lon_grid},
        dims=("lat", "lon")
    )

    lon_t = traj_df["lon"].to_numpy(float)
    lat_t = traj_df["lat"].to_numpy(float)
    z_t   = traj_df["z"].to_numpy(float)

    # 插值海底高程（双线性）
    z_bathy = da.interp(
        lon=xr.DataArray(lon_t, dims="p"),
        lat=xr.DataArray(lat_t, dims="p"),
        method="linear"
    ).to_numpy()

    # 净空：正值表示“离海底还有多少米”，负值表示“已经穿进海底”
    clearance = z_t - z_bathy

    min_clear = float(np.nanmin(clearance))
    imin = int(np.nanargmin(clearance))
    hit = bool(min_clear <= float(safety_margin))

    return {
        "hit": hit,
        "min_clearance_m": min_clear,
        "argmin_index": imin,
        "z_traj_at_min": float(z_t[imin]),
        "z_bathy_at_min": float(z_bathy[imin]),
        "clearance": clearance,
        "z_bathy": z_bathy
    }


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # ====== 按你的目录结构 ======
    ETOPO_PATH = os.path.join(ROOT, "data", "etopo", "ETOPO_2022_v1_60s_N90W180_bed.nc")
    TRAJ_CSV   = os.path.join(ROOT, "output", "one_trajectory.csv")
    OUT_PNG    = os.path.join(ROOT, "output", "trajectory_3d_over_bathy.png")
    OUT_BATHY_NC = os.path.join(ROOT, "output", "bathy_cropped.nc")  # 可选输出：裁剪后的海底地形
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

    if not os.path.exists(TRAJ_CSV):
        raise FileNotFoundError(f"找不到轨迹 CSV：{TRAJ_CSV}")

    traj = pd.read_csv(TRAJ_CSV, encoding="utf-8-sig")
    for c in ["lon", "lat", "z"]:
        if c not in traj.columns:
            raise ValueError(f"轨迹 CSV 缺少列 {c}，当前列：{list(traj.columns)}")

    # ====== 轨迹经纬度 sanity check：防止弧度误当度 ======
    lon_abs_max = float(np.nanmax(np.abs(traj["lon"].to_numpy(float))))
    lat_abs_max = float(np.nanmax(np.abs(traj["lat"].to_numpy(float))))
    if lon_abs_max <= 3.5 and lat_abs_max <= 3.5:
        print("[WARN] traj lon/lat 看起来像弧度(rad)，自动转换为度(deg)")
        traj["lon"] = np.degrees(traj["lon"].to_numpy(float))
        traj["lat"] = np.degrees(traj["lat"].to_numpy(float))

    # 局部坐标原点：用轨迹起点（你也可以改成均值点）
    lon0 = float(traj["lon"].iloc[0])
    lat0 = float(traj["lat"].iloc[0])

    # ====== 自动生成合适大小裁剪框（米） ======
    bbox_xy = auto_bbox_from_traj_xy(
        traj, lon0, lat0,
        pad_ratio=2.5,     # 你轨迹是 km 级，这个一般很合适
        pad_min_m=2500.0,  # 最少留 2.5km 上下文
        pad_max_m=20000.0
    )
    print(f"[INFO] traj extent (m): x=({bbox_xy['traj_x_min']:.1f},{bbox_xy['traj_x_max']:.1f}), "
          f"y=({bbox_xy['traj_y_min']:.1f},{bbox_xy['traj_y_max']:.1f}), pad={bbox_xy['pad_m']:.1f}m")
    print(f"[INFO] crop bbox (m): x=({bbox_xy['x_min']:.1f},{bbox_xy['x_max']:.1f}), "
          f"y=({bbox_xy['y_min']:.1f},{bbox_xy['y_max']:.1f})")

    # ====== 米裁剪框 -> 经纬裁剪范围 ======
    lon_min, lon_max, lat_min, lat_max = bbox_xy_to_ll(bbox_xy, lon0, lat0)
    print(f"[INFO] crop bbox (deg): lon=({lon_min:.6f},{lon_max:.6f}), lat=({lat_min:.6f},{lat_max:.6f})")

    print("[INFO] loading ETOPO subset ...")
    lon_g, lat_g, Z = load_etopo_subset(ETOPO_PATH, lon_min, lon_max, lat_min, lat_max, engine="netcdf4")

    res = check_bottom_collision(lon_g, lat_g, Z, traj, safety_margin=20.0)  # 例如设 20m 安全间隙
    print("[CHECK] hit=", res["hit"])
    print("[CHECK] min clearance (m)=", res["min_clearance_m"])
    print("[CHECK] at idx=", res["argmin_index"],
          "z_traj=", res["z_traj_at_min"],
          "z_bathy=", res["z_bathy_at_min"])

    # ====== 可选：重采样到 200m 间距（用于统一网格、贴合后续栅格仿真） ======
    # 注意：ETOPO 60s 本身 ~1-2km，200m 是插值网格，不会凭空增加细节，但对统一网格很有用
    print("[INFO] resampling bathy grid to ~200m ...")
    lon_g, lat_g, Z = resample_bathy_to_meter_grid(
        lon_g, lat_g, Z,
        lon0=lon0, lat0=lat0,
        dx=200.0, dy=200.0,
        method="linear",
        max_cells=2_000_000
    )

    print("[INFO] plotting 3D ...")
    plot_3d_bathy_and_traj(
        lon_g, lat_g, Z,
        traj_df=traj,
        lon0=lon0, lat0=lat0,
        out_png=OUT_PNG,
        bbox_xy=bbox_xy,    # 锁定视野
        downsample=2,       # 200m 网格下，2~4 通常比较流畅
        z_exag=60,          # 你那片海域 std 不大时，夸张一点纹理更明显
        elev=22, azim=-55
    )

    print("[OK] saved:", OUT_PNG)


if __name__ == "__main__":
    main()
