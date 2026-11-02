"""Export the README charts to docs/images/, in a light and a dark version.

uv run --group docs python scripts/export_gallery.py

Runs the same batch as run_demo.py, so every number in a chart comes from the
engine. Colours are the qk palette (Tailwind 600 for light, 400 for dark).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from derivatives_risk_engine import pnl_explain
from derivatives_risk_engine.engine import run, summary_tables

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

THEMES = {
    "light": {
        "bg": "#ffffff",
        "text": "#0f172a",
        "muted": "#64748b",
        "grid": "#e2e8f0",
        "accent": "#2563eb",
        "danger": "#dc2626",
        "success": "#059669",
        "warning": "#d97706",
        "purple": "#7c3aed",
        "neutral": "#94a3b8",
        "zone_g": "#d1fae5", "on_strong": "#ffffff",
        "zone_a": "#fef3c7",
        "zone_r": "#fee2e2",
    },
    "dark": {
        "bg": "#0d1117",
        "text": "#e6edf3",
        "muted": "#8b949e",
        "grid": "#30363d",
        "accent": "#60a5fa",
        "danger": "#f87171",
        "success": "#34d399",
        "warning": "#fbbf24",
        "purple": "#a78bfa",
        "neutral": "#6e7681",
        "zone_g": "#0f2e24", "on_strong": "#0d1117",
        "zone_a": "#2e2610",
        "zone_r": "#3a1616",
    },
}
M = 1e6


def desk_name(d: str) -> str:
    return {"FX": "FX", "FIRM": "Firm"}.get(d, d.title())


def style(t: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 10.5,
            "figure.facecolor": t["bg"],
            "axes.facecolor": t["bg"],
            "savefig.facecolor": t["bg"],
            "text.color": t["text"],
            "axes.labelcolor": t["text"],
            "axes.edgecolor": t["grid"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "axes.grid": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "legend.frameon": False,
        }
    )


def save(fig, name: str, theme: str) -> None:
    fig.savefig(
        OUT / f"{name}-{theme}.png", dpi=200, bbox_inches="tight", pad_inches=0.3
    )
    plt.close(fig)


def pnl_distribution(firm: np.ndarray, var99: float, es975: float, t: dict):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = firm / M
    bins = np.linspace(x.min(), x.max(), 60)
    tail = x <= -var99 / M
    ax.hist(x[~tail], bins=bins, color=t["accent"], alpha=0.85, label="daily P&L")
    ax.hist(x[tail], bins=bins, color=t["danger"], alpha=0.95, label="beyond VaR 99%")
    for v, lab, ls in ((var99, "VaR 99%", "-"), (es975, "ES 97.5%", "--")):
        ax.axvline(-v / M, color=t["danger"], ls=ls, lw=1.4)
        ax.annotate(
            f"{lab}\n{v / M:.1f}m",
            (-v / M, ax.get_ylim()[1] * 0.92),
            xytext=(6 if ls == "-" else -6, 0),
            textcoords="offset points",
            ha="left" if ls == "-" else "right",
            va="top",
            color=t["danger"],
            fontsize=10,
            fontweight="semibold",
        )
    ax.set_title("Firm daily P&L, full revaluation over 500 synthetic days")
    ax.set_xlabel("P&L (USD m)")
    ax.set_ylabel("days")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return fig


def var_diversification(var_table: pd.DataFrame, t: dict):
    v = var_table["VaR 99%"] / M
    desks = v.drop(["Sum of desks", "FIRM"]).sort_values()
    labels = [*map(desk_name, desks.index), "Sum of desks", "Firm"]
    vals = [*desks.values, v["Sum of desks"], v["FIRM"]]
    colors = [t["accent"]] * len(desks) + [t["neutral"], t["success"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, val in zip(y, vals):
        ax.text(val + 1, yi, f"{val:.1f}m", va="center", fontsize=10, color=t["text"])
    gap = v["Sum of desks"] - v["FIRM"]
    ax.plot([v["FIRM"] + 9, v["Sum of desks"]], [y[-1], y[-1]], color=t["success"],
            lw=1.2, ls="--")
    ax.plot([v["Sum of desks"]] * 2, [y[-1], y[-2] + 0.31], color=t["success"], lw=1.2,
            ls="--")
    ax.text(v["Sum of desks"] + 2, y[-1],
            f"−{gap:.1f}m ({gap / v['Sum of desks']:.0%})\ndiversification",
            va="center", fontsize=10, color=t["success"], fontweight="semibold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, v["Sum of desks"] * 1.3)
    ax.set_xlabel("1-day VaR 99%, USD m")
    ax.set_title("VaR by desk: desks do not add up to the firm")
    ax.grid(axis="y", visible=False)
    return fig


def stress_heatmap(stress: pd.DataFrame, t: dict):
    d = stress / M
    d.columns = [desk_name(c) for c in d.columns]
    cmap = LinearSegmentedColormap.from_list("pl", [t["danger"], t["bg"], t["accent"]])
    norm = TwoSlopeNorm(vcenter=0, vmin=-200, vmax=400)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.imshow(d.clip(-200, 400).to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            val = d.iat[i, j]
            txt = "0" if abs(val) < 0.05 else f"{val:+,.0f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color=t["on_strong"] if abs(norm(np.clip(val, -200, 400)) - 0.5) > 0.35 else t["text"],
                fontweight="semibold" if j == d.shape[1] - 1 else None,
            )
    ax.set_xticks(range(d.shape[1]), d.columns)
    ax.set_yticks(range(d.shape[0]), d.index)
    ax.axvline(d.shape[1] - 1.5, color=t["text"], lw=1)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Stress P&L by desk, USD m (blue gain, red loss)", pad=28)
    return fig


OFFSETS = {"CREDIT": (0, 7), "FX": (0, -14)}


def pla_zones(pla: pd.DataFrame, t: dict):
    models = {
        "full": "Full risk factors",
        "beta": "One factor per curve, fitted betas",
        "naive": "Anchor copied onto the curve",
    }
    g, r = pnl_explain.GREEN, pnl_explain.RED
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    zc = {"green": t["success"], "amber": t["warning"], "red": t["danger"]}
    for ax, (k, title) in zip(axes, models.items()):
        ax.set_xlim(0, 0.25)
        ax.set_ylim(0.35, 1.08)
        ax.axvspan(0, 0.25, color=t["zone_a"], zorder=0)
        ax.fill_between([0, g["ks"]], g["spearman"], 1.08, color=t["zone_g"], zorder=0)
        ax.axvspan(r["ks"], 0.25, color=t["zone_r"], zorder=0)
        ax.fill_between([0, 0.25], 0.35, r["spearman"], color=t["zone_r"], zorder=0)
        ax.grid(False)
        for desk, row in pla.iterrows():
            x, y = row[f"{k}_ks"], row[f"{k}_spearman"]
            ax.scatter(
                x,
                y,
                s=46,
                color=zc[row[f"{k}_zone"]],
                edgecolor=t["bg"],
                lw=1,
                zorder=3,
            )
            if k != "full":
                ax.annotate(
                    desk_name(desk),
                    (x, y),
                    xytext=OFFSETS.get(desk, (7, -3)),
                    ha="center" if desk in OFFSETS else "left",
                    textcoords="offset points",
                    fontsize=9,
                    color=t["text"],
                )
        if k == "full":
            for zx, zy, lab, zk in ((0.045, 0.83, "green", "success"),
                                    (0.045, 0.735, "amber", "warning"),
                                    (0.18, 0.55, "red", "danger")):
                ax.text(zx, zy, lab, ha="center", fontsize=9, color=t[zk],
                        fontweight="semibold")
            ax.annotate(
                "all 5 desks",
                (0.014, 1.0),
                xytext=(0.05, 0.9),
                fontsize=9,
                color=t["text"],
                arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 0.8},
            )
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("KS distance")
    axes[0].set_ylabel("Spearman correlation")
    fig.suptitle(
        "FRTB P&L attribution test (MAR32), 250 out-of-sample days",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def frtb_capital(by_class: pd.DataFrame, total: float, t: dict):
    names = {
        "CMDTY": "Commodity",
        "CSR": "Credit spread",
        "GIRR": "Interest rate",
        "FX": "FX",
        "EQ": "Equity",
    }
    d = by_class.set_index("risk_class") if "risk_class" in by_class else by_class
    d = d.sort_values("total") / M
    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for col, c in (
        ("delta", t["accent"]),
        ("vega", t["purple"]),
        ("curvature", t["warning"]),
    ):
        ax.barh(y, d[col], left=left, color=c, height=0.62, label=col)
        left += d[col].to_numpy()
    for yi, val in zip(y, d["total"]):
        ax.text(val + 6, yi, f"{val:,.0f}m", va="center", fontsize=10, color=t["text"])
    ax.set_yticks(y, [names.get(i, i) for i in d.index])
    ax.set_xlim(0, d["total"].max() * 1.15)
    ax.set_xlabel("USD m")
    ax.set_title(f"FRTB standardised capital by risk class (firm {total / M:,.0f}m)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=3)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = run()
    tables = summary_tables(r)
    firm = r.trade_pnl.sum().to_numpy()
    var99, es975 = tables["var"].loc["FIRM", ["VaR 99%", "ES 97.5%"]]
    for name, t in THEMES.items():
        style(t)
        save(pnl_distribution(firm, var99, es975, t), "pnl-distribution", name)
        save(var_diversification(tables["var"], t), "var-diversification", name)
        save(stress_heatmap(tables["stress"], t), "stress", name)
        save(pla_zones(tables["pla"], t), "pla-zones", name)
        save(
            frtb_capital(tables["frtb_by_class"], r.frtb_book.by_scenario.loc[r.frtb_book.binding_scenario, "total"], t),
            "frtb-capital",
            name,
        )
    print(f"wrote {len(list(OUT.glob('*.png')))} images to {OUT}")


if __name__ == "__main__":
    main()
