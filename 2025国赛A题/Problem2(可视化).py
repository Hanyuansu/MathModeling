import os
from typing import List, Tuple
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
def plot_cover_timeline(
    intervals: List[Tuple[float, float]],
    *,
    xlim: Tuple[float, float],
    title: str = None,
    title_prefix: str = "M1",
    show_total: bool = False,
    figsize=(8.6, 2.0),
    bg_color="#dddddd",
    fg_color="tab:red",
    bg_lw=10,
    fg_lw=10,
    xlabel="t / s",
    legend_loc="upper right",
    save_path: str | None = None,
    dpi=180,
    transparent=False,
    pad_inches=0.05,
    show=True,
    close=False,
    return_fig_ax: bool = False
):


    t0, t1 = xlim
    total_cover = float(sum(max(0.0, b - a) for (a, b) in intervals))

    if title is None:
        title = f"{title_prefix}  总遮蔽时长={total_cover:.3f} s" if show_total else f"{title_prefix}"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    y = 1.0

    ax.hlines(y, t0, t1, color=bg_color, lw=bg_lw, label="有效窗口")

    for (a, b) in intervals:
        aa = max(a, t0)
        bb = min(b, t1)
        if bb > aa:
            ax.hlines(y, aa, bb, color=fg_color, lw=fg_lw, label="遮掩段")

    ax.set_xlim(t0, t1)
    ax.set_ylim(0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc=legend_loc, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, transparent=transparent, pad_inches=pad_inches)
        print(f"[OK] 已保存：{save_path}")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    if return_fig_ax:
        return fig, ax

if __name__ == "__main__":
    out_dir = os.path.join("result", "Problem2_result")
    t_burst_TS = 0.869068730035006
    intervals_TS = [(0.869068730035006, 5.929068730035007)]
    plot_cover_timeline(
        intervals_TS,
        xlim=(0, t_burst_TS + 20.0),
        title="M1  总遮蔽时长=5.060 s",
        figsize=(8.6, 2.0),
        bg_color="#dddddd",
        fg_color="tab:red",
        bg_lw=10,
        fg_lw=10,
        save_path=os.path.join(out_dir, "遮掩时长.png"),
        dpi=180,
        transparent=False,
        pad_inches=0.05,
        show=True,
        close=False
    )
