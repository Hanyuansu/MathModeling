import os
from typing import List, Tuple, Sequence

def plot_cover_timeline(
    intervals: List[Tuple[float, float]],
    *,
    xlim: Tuple[float, float],
    vlines_drop: Sequence[float] | None = None,
    vlines_burst: Sequence[float] | None = None,
    title: str = None,
    figsize=(8.6, 2.0),
    bg_color="#dddddd",
    fg_color="tab:red",
    drop_color="tab:blue",
    burst_color="tab:orange",
    save_path: str | None = None,
    dpi=180,
    transparent=False,
    pad_inches=0.05,
    show=True,
    close=False,
):
    import matplotlib.pyplot as plt

    try:
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    t0, t1 = xlim
    total_cover = sum(b - a for (a, b) in intervals)
    if title is None:
        title = f"M1  总遮蔽时长={total_cover:.2f} s"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    y = 1.0

    ax.hlines(y, t0, t1, color=bg_color, lw=10, label="有效窗口")

    for (a, b) in intervals:
        if b > a:
            ax.hlines(y, a, b, color=fg_color, lw=10, label="遮掩段")

    if vlines_drop:
        for i, x in enumerate(vlines_drop):
            if t0 <= x <= t1:
                ax.axvline(x, color=drop_color, ls="--", lw=1.8,
                           label="投放" if i == 0 else None)

    if vlines_burst:
        for i, x in enumerate(vlines_burst):
            if t0 <= x <= t1:
                ax.axvline(x, color=burst_color, ls=":", lw=1.8,
                           label="起爆" if i == 0 else None)

    ax.set_xlim(t0, t1)
    ax.set_ylim(0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("t / s")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(),
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, transparent=transparent,
                    pad_inches=pad_inches, bbox_inches="tight")
        print(f"[OK] 已保存：{save_path}")
    if show:
        plt.show()
    if close:
        plt.close(fig)

if __name__ == "__main__":
    out_dir = os.path.join("result", "Problem4_result")

    intervals_Q4 = [(0.88, 5.9), (15.0, 17.98), (33.0, 35.7)]
    drops  = [0.026, 7.025, 28.739]
    bursts = [0.869, 15.0, 33.0]

    t0, t1 = 0.0, 36.0
    xlim_Q4 = (t0, t1)

    plot_cover_timeline(
        intervals_Q4,
        xlim=xlim_Q4,
        vlines_drop=drops,
        vlines_burst=bursts,
        title="M1  总遮蔽时长=10.76 s",
        save_path=os.path.join(out_dir, "遮掩时长.png"),
        show=True
    )
