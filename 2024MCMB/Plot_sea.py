import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.cm import ScalarMappable


def load_etopo_subset(etopo_path, lon_min, lon_max, lat_min, lat_max, engine="netcdf4"):
    """
    读取 ETOPO2022 的子区域
    ETOPO2022 60s 文件常见结构：coords = lon/lat, var = z (m)
    """
    if not os.path.exists(etopo_path):
        raise FileNotFoundError(f"ETOPO 文件不存在：{etopo_path}")

    ds = xr.open_dataset(etopo_path, decode_times=False, engine=engine)

    # 注意：ETOPO 的 lon 通常是 [-180, 180]
    z = ds["z"].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)).load()
    ds.close()

    lon = z["lon"].values.astype(float)
    lat = z["lat"].values.astype(float)
    Z = z.values.astype(float)

    # 确保 lon/lat 单调递增（少数文件可能不是严格升序）
    if not np.all(np.diff(lon) > 0):
        ix = np.argsort(lon)
        lon = lon[ix]
        Z = Z[:, ix]
    if not np.all(np.diff(lat) > 0):
        iy = np.argsort(lat)
        lat = lat[iy]
        Z = Z[iy, :]

    Z[~np.isfinite(Z)] = np.nan
    return lon, lat, Z


def plot_3d_bathy_like_example(
    lon, lat, Z,
    out_png,
    title="Ionian Sea Bathymetry (ETOPO2022, 3D)",
    downsample=3,
    add_sea_surface=True,
    sea_alpha=0.25,
    elev=18,
    azim=-65,
    zlim=(-5200, 3200),
    use_shading=True
):
    """
    画出类似你示例图的 3D 拓扑：
    - x=longitude(°), y=latitude(°), z=depth/elevation(m)
    - colorbar
    - z=0 海面平面（可选）
    """
    Lon, Lat = np.meshgrid(lon, lat)

    # 降采样（ETOPO 60s 很密，直接画会非常慢）
    Lon2 = Lon[::downsample, ::downsample]
    Lat2 = Lat[::downsample, ::downsample]
    Z2 = Z[::downsample, ::downsample]

    # masked array：3D surface 对 NaN 更稳定
    Zm = np.ma.masked_invalid(Z2)

    # 颜色映射范围：按数据自动，也可手动
    z_valid = Z2[np.isfinite(Z2)]
    if z_valid.size == 0:
        raise RuntimeError("该区域没有有效 z 数据（全是 NaN），检查 bbox 是否正确")

    vmin = float(np.nanpercentile(z_valid, 1))
    vmax = float(np.nanpercentile(z_valid, 99))

    # 做一点保护：避免 vmin==vmax
    if abs(vmax - vmin) < 1e-6:
        vmin = float(np.nanmin(z_valid))
        vmax = float(np.nanmax(z_valid))

    cmap = plt.cm.terrain

    # 光照增强纹理（更像“山脊”那种效果）
    if use_shading:
        ls = LightSource(azdeg=315, altdeg=45)
        Z_for_shade = np.nan_to_num(Z2, nan=float(np.nanmin(z_valid)))
        facecolors = ls.shade(Z_for_shade, cmap=cmap, vert_exag=1.0, blend_mode="soft")
    else:
        facecolors = None

    fig = plt.figure(figsize=(14, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # 画地形/海底曲面
    if facecolors is not None:
        surf = ax.plot_surface(
            Lon2, Lat2, Zm,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False
        )
    else:
        surf = ax.plot_surface(
            Lon2, Lat2, Zm,
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidth=0,
            antialiased=True
        )

    # 画海面平面 z=0（像你图里那块“蓝色平面”）
    if add_sea_surface:
        z0 = np.zeros_like(Lon2, dtype=float)
        ax.plot_surface(
            Lon2, Lat2, z0,
            color=(0.1, 0.25, 0.6),
            alpha=sea_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )

    # 坐标与视角
    ax.set_title(title, pad=12)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_zlabel("depth")

    ax.view_init(elev=elev, azim=azim)

    # zlim 用你示例图风格（海底到 -5000，陆地到 3000）
    if zlim is not None:
        ax.set_zlim(float(zlim[0]), float(zlim[1]))

    # grid 更像论文（淡一点）
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
    ax.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
    ax.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)

    # colorbar：如果用 facecolors（shading），plot_surface 不会自动带 mappable
    sm = ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("elevation / depth (m)")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # 你的 ETOPO 文件路径
    ETOPO_PATH = os.path.join(ROOT, "data", "etopo", "ETOPO_2022_v1_60s_N90W180_bed.nc")

    OUT_DIR = os.path.join(ROOT, "output")
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_PNG = os.path.join(OUT_DIR, "ionian_bathy_3d_topo.png")

    # Ionian Sea bbox（你一直用的范围）
    LON_MIN, LON_MAX = 13.0, 22.5
    LAT_MIN, LAT_MAX = 33.0, 40.5

    print("[INFO] loading ETOPO subset ...")
    lon, lat, Z = load_etopo_subset(ETOPO_PATH, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, engine="netcdf4")

    print("[INFO] plotting 3D bathymetry ...")
    plot_3d_bathy_like_example(
        lon, lat, Z,
        out_png=OUT_PNG,
        title="Ionian Sea Bathymetry (ETOPO2022, 3D)",
        downsample=3,          # 想更细：2（更慢）；更快：4/5
        add_sea_surface=True,  # 加海面平面
        sea_alpha=0.25,
        elev=18, azim=-65,     # 视角接近你图
        zlim=(-5200, 3200),
        use_shading=True
    )

    print("[OK] saved:", OUT_PNG)


if __name__ == "__main__":
    main()
