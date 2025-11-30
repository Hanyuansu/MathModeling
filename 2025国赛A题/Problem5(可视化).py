import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def plot_cover_timeline_multi(per_missile: List[Dict[str, Any]],
                              *,
                              drops: List[float],
                              bursts: List[float],
                              title: str = None,
                              figsize=(9, 4.5),
                              save_path=None,
                              dpi=180):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    y_base = 1.0
    y_gap = 0.8
    colors = ["tab:red", "tab:green", "tab:purple"]

    for idx, info in enumerate(per_missile):
        y = y_base - idx * y_gap
        missile = info["missile"]
        T_hit = info["T_hit"]
        intervals = info["intervals"]

        t0 = min(a for a, _ in intervals)
        ax.hlines(y, t0, T_hit, color="#dddddd", lw=8,
                  label="Vaild Window" if idx == 0 else None)

        mid_win = (t0 + T_hit) / 2
        ax.text(mid_win, y + 0.15, f"{missile} ({info['cover_s']:.2f}s)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

        for (a, b) in intervals:
            ax.hlines(y, a, b, color=colors[idx % len(colors)], lw=8,
                      label=f"{missile} Covering Section" )

        for j, d in enumerate(drops):
            if t0 <= d <= T_hit:
                ax.axvline(d, color="tab:blue", ls="--", lw=1.0, alpha=0.6,
                           label="Drop" if (idx == 0 and j == 0) else None)

        for j, b in enumerate(bursts):
            if t0 <= b <= T_hit:
                ax.axvline(b, color="tab:orange", ls=":", lw=1.0, alpha=0.6,
                           label="Brust" if (idx == 0 and j == 0) else None)

    ax.set_xlabel("t / s")
    if title:
        ax.set_title(title)

    ax.set_yticks([])
    ax.set_ylim(y_base - (len(per_missile) - 1) * y_gap - 0.5, y_base + 0.6)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(),
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] 已保存：{save_path}")
    plt.show()


if __name__ == "__main__":
    out_dir = os.path.join("result", "Problem5_result")

    per_missile = [
        {"missile": "M1", "T_hit": 66.999, "cover_s": 20.1,
         "intervals": [(3.75, 9.81), (15.045, 23.895), (30.75, 35.895)]},
        {"missile": "M2", "T_hit": 63.75, "cover_s": 4.41,
         "intervals": [(10.935, 15.33)]},
        {"missile": "M3", "T_hit": 60.366, "cover_s": 3.72,
         "intervals": [(23.25, 26.955)]},
    ]

    drops  = [12.315, 10.725, 22.995, 24.075, 1.395, 3.045, 0.0, 2.46, 9.24,
              10.875, 1.785, 5.58, 11.085, 5.58, 7.095]
    bursts = [14.977, 10.925, 23.246, 25.047, 5.598, 15.045, 3.739, 2.932,
              9.688, 11.14, 1.997, 6.522, 11.367, 6.272, 7.417]

    plot_cover_timeline_multi(
        per_missile,
        drops=drops,
        bursts=bursts,
        title=" Total obscuration duration=28.23 s",
        save_path=os.path.join(out_dir, "遮掩时长.png")
    )
