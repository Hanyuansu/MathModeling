import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import os
from typing import List, Tuple, Sequence

def plot_cover_timeline(
    intervals: List[Tuple[float, float]],
    *,
    xlim: Tuple[float, float],
    vlines_drop: Sequence[float] | None = None,
    vlines_burst: Sequence[float] | None = None,
    show_drop_labels: bool = True,
    show_burst_labels: bool = True,
    drop_label_prefix: str = "投放",
    burst_label_prefix: str = "起爆",
    drop_linestyle: str = "--",
    burst_linestyle: str = ":",
    drop_color: str = "tab:blue",
    burst_color: str = "tab:orange",

    drop_text_rotation: float = 90.0,
    burst_text_rotation: float = 90.0,
    text_y: float = 1.19,
    text_fontsize: int = 9,

    title: str | None = None,
    title_prefix: str = "M1",
    show_total: bool = False,
    figsize=(8.6, 2.0),
    bg_color="#dddddd",
    fg_color="tab:red",
    bg_lw=10,
    fg_lw=10,
    xlabel="t / s",

    legend_outside: bool = True,
    legend_loc_inside: str = "upper right",
    legend_bbox: Tuple[float, float] = (1.02, 1.0),
    legend_loc_outside: str = "upper left",

    save_path: str | None = None,
    dpi=180,
    transparent=False,
    pad_inches=0.05,
    show=True,
    close=False,
    return_fig_ax: bool = False,
):

    import matplotlib.pyplot as plt

    try:
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    t0, t1 = xlim
    total_cover = float(sum(max(0.0, b - a) for (a, b) in intervals))
    if title is None:
        title = f"{title_prefix}  总遮蔽时长={total_cover:.3f} s" if show_total else f"{title_prefix}"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    y = 1.0

    ax.hlines(y, t0, t1, color=bg_color, lw=bg_lw, label="Valid Window")

    for (a, b) in intervals:
        aa, bb = max(a, t0), min(b, t1)
        if bb > aa:
            ax.hlines(y, aa, bb, color=fg_color, lw=fg_lw, label="Covering Section")


    def _draw_vlines(xs: Sequence[float] | None, ls: str, col: str, label: str):
        if not xs:
            return []
        xs_in = [x for x in xs if (t0 <= x <= t1)]
        if not xs_in:
            return []
        h0, h1 = 0.84, 1.16
        first = True
        for x in xs_in:
            ax.vlines(x, h0, h1, linestyles=ls, colors=col, lw=1.8,
                      label=(label if first else None))
            first = False
        return sorted(xs_in)

    xs_drop_in  = _draw_vlines(vlines_drop,  drop_linestyle,  drop_color,  "Drop")
    xs_burst_in = _draw_vlines(vlines_burst, burst_linestyle, burst_color, "Brust")

    ax.set_xlim(t0, t1)
    ax.set_ylim(0.8, 1.30)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))

    if legend_outside:
        ax.legend(
            uniq.values(), uniq.keys(),
            loc=legend_loc_outside, bbox_to_anchor=legend_bbox,
            borderaxespad=0.0, fontsize=9
        )
    else:
        ax.legend(uniq.values(), uniq.keys(), loc=legend_loc_inside, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, transparent=transparent,
                    pad_inches=pad_inches, bbox_inches="tight")
        print(f"[OK] 已保存：{save_path}")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    if return_fig_ax:
        return fig, ax

if __name__ == "__main__":
    out_dir = os.path.join("result", "Problem3_result")

    intervals_Q3 = [(1.2, 7.36)]

    drops  = [0.0, 1.0, 25.654]
    bursts = [0.2, 1.2, 25.854328299370056]


    t0 = drops[0]
    xlim_Q3 = (t0, t0 + 20.0)

    plot_cover_timeline(
        intervals_Q3,
        xlim=xlim_Q3,
        vlines_drop=drops,
        vlines_burst=bursts,
        title="M1  Total obscuration duration=6.16 s",
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
