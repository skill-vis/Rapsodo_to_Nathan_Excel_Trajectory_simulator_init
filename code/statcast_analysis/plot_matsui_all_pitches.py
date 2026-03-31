"""
Matsui 2025 全投球の球種別分布 + 95%楕円
japanese_pitchers_sorted.png と同じスタイル・軸範囲
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from pybaseball import statcast

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FT2M = 0.3048
CACHE = os.path.join(_SCRIPT_DIR, "matsui_2025_all.json")

# ===================================================================
# Fetch / cache
# ===================================================================
if os.path.exists(CACHE):
    print("Loading cached data ...")
    with open(CACHE) as f:
        pitch_data = json.load(f)
else:
    print("Fetching Matsui 2025 all pitches ...")
    df = statcast(start_dt="2025-03-20", end_dt="2025-10-01")
    df_d = df[df["pitcher"] == 673513].copy()
    print(f"Total: {len(df_d)} pitches")
    print(df_d["pitch_type"].value_counts())

    # Strike zone: average per batter
    sz_top = df_d["sz_top"].mean()
    sz_bot = df_d["sz_bot"].mean()

    pitch_data = {"sz_top": sz_top, "sz_bot": sz_bot, "types": {}}
    for pt, grp in df_d.groupby("pitch_type"):
        if len(grp) < 5:
            continue
        x = grp["plate_x"].dropna().tolist()
        z = grp["plate_z"].dropna().tolist()
        pitch_data["types"][pt] = {"x": x, "z": z, "n": len(x)}
    with open(CACHE, "w") as f:
        json.dump(pitch_data, f)
    print(f"Cached to {CACHE}")

# ===================================================================
# Color map for pitch types (MLB standard-ish)
# ===================================================================
PITCH_COLORS = {
    "FF": ("#d62728", "4-Seam"),     # red
    "SI": ("#ff7f0e", "Sinker"),     # orange
    "FC": ("#8c564b", "Cutter"),     # brown
    "SL": ("#2ca02c", "Slider"),     # green
    "CU": ("#1f77b4", "Curveball"),  # blue
    "CH": ("#9467bd", "Changeup"),   # purple
    "FS": ("#e377c2", "Splitter"),   # pink
    "FO": ("#e377c2", "Forkball"),   # pink
    "ST": ("#17becf", "Sweeper"),    # cyan
    "KC": ("#aec7e8", "Knuckle-CV"), # light blue
    "SV": ("#98df8a", "Slurve"),     # light green
}

# ===================================================================
# Draw
# ===================================================================
types = pitch_data["types"]
sz_top_m = pitch_data["sz_top"] * FT2M
sz_bot_m = pitch_data["sz_bot"] * FT2M

# Sort by count descending
sorted_types = sorted(types.items(), key=lambda kv: -kv[1]["n"])
n_types = len(sorted_types)

# Layout: use same figure size ratio as 9-panel
ncols = min(n_types, 3)
nrows = (n_types + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6.3 * nrows))
if nrows == 1 and ncols == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = axes[np.newaxis, :]
elif ncols == 1:
    axes = axes[:, np.newaxis]

# Also draw an "ALL" overlay panel — put it at the end
# Actually, let's add one more panel for all pitches overlaid
total_panels = n_types + 1  # +1 for ALL overlay
ncols_all = min(total_panels, 3)
nrows_all = (total_panels + ncols_all - 1) // ncols_all

fig, axes = plt.subplots(nrows_all, ncols_all, figsize=(15, 6.3 * nrows_all))
if nrows_all == 1:
    axes = axes[np.newaxis, :] if ncols_all > 1 else np.array([[axes]])

# Axis limits matching japanese_pitchers_sorted.png
XLIM = (-0.75, 0.75)
YLIM = (0, 1.85)

def draw_strike_zone(ax):
    hw = 0.708 * FT2M
    sz = patches.Rectangle((-hw, sz_bot_m), 2 * hw, sz_top_m - sz_bot_m,
                            linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--")
    ax.add_patch(sz)
    plate_x = [0, hw, hw, -hw, -hw, 0]
    plate_y = np.array([0, 0.35, 0.71, 0.71, 0.35, 0]) * FT2M - 0.15
    ax.plot(plate_x, plate_y, "k-", linewidth=1)

def draw_ellipse_95(ax, x, z, color):
    if len(x) < 5:
        return
    data = np.column_stack([x, z])
    mean = data.mean(axis=0)
    cov = np.cov(data.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    chi2_95 = 5.991
    w = 2 * np.sqrt(evals[0] * chi2_95)
    h = 2 * np.sqrt(evals[1] * chi2_95)
    ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                  edgecolor=color, facecolor="none", linewidth=2, linestyle="-", clip_on=True)
    ax.add_patch(ell)
    ax.plot(mean[0], mean[1], "o", color=color, markersize=5, zorder=5,
            markeredgecolor="black", markeredgewidth=0.5)

# --- Per-type panels ---
for idx, (pt, td) in enumerate(sorted_types):
    row, col = divmod(idx, ncols_all)
    ax = axes[row][col]
    x = np.array(td["x"]) * FT2M
    z = np.array(td["z"]) * FT2M
    n = td["n"]
    color, fullname = PITCH_COLORS.get(pt, ("#999999", pt))

    draw_strike_zone(ax)
    ax.scatter(x, z, alpha=0.25, s=10, c=color, edgecolors=color, linewidth=0.2)
    draw_ellipse_95(ax, x, z, color)

    # Zone rate
    x_ft = np.array(td["x"])
    z_ft = np.array(td["z"])
    in_zone = np.sum((x_ft >= -0.708) & (x_ft <= 0.708) &
                      (z_ft >= pitch_data["sz_bot"]) & (z_ft <= pitch_data["sz_top"]))
    zone_pct = in_zone / n * 100

    ax.text(0, 0.03, f"Zone: {zone_pct:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.85))

    ax.set_title(f"{fullname} ({pt})  n={n}", fontsize=10, fontweight="bold", color=color)
    ax.set_aspect("equal")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_adjustable("box")
    ax.grid(True, alpha=0.2)
    if col == 0:
        ax.set_ylabel("Vertical (m)", fontsize=9)
    if row == nrows_all - 1:
        ax.set_xlabel("Horizontal (m)", fontsize=9)

# --- ALL overlay panel ---
all_idx = n_types
row, col = divmod(all_idx, ncols_all)
ax = axes[row][col]
draw_strike_zone(ax)
for pt, td in sorted_types:
    x = np.array(td["x"]) * FT2M
    z = np.array(td["z"]) * FT2M
    color, fullname = PITCH_COLORS.get(pt, ("#999999", pt))
    ax.scatter(x, z, alpha=0.15, s=8, c=color, edgecolors=color, linewidth=0.2, label=f"{pt} ({td['n']})")
    draw_ellipse_95(ax, x, z, color)
total_n = sum(td["n"] for _, td in sorted_types)
ax.set_title(f"ALL PITCHES  n={total_n}", fontsize=10, fontweight="bold")
ax.set_aspect("equal")
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_adjustable("box")
ax.grid(True, alpha=0.2)
ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
if col == 0:
    ax.set_ylabel("Vertical (m)", fontsize=9)
ax.set_xlabel("Horizontal (m)", fontsize=9)

# Hide unused panels
for idx in range(all_idx + 1, nrows_all * ncols_all):
    row, col = divmod(idx, ncols_all)
    axes[row][col].set_visible(False)

fig.suptitle("Matsui 2025 All Pitch Types — Location Distribution & 95% Ellipse",
             fontsize=13, fontweight="bold", y=1.0)
plt.tight_layout()
# Force ALL axes to same width as individual panels
for r in range(nrows_all):
    for c in range(ncols_all):
        if axes[r][c].get_visible():
            pos = axes[0][0].get_position()
            w = pos.x1 - pos.x0
            cur = axes[r][c].get_position()
            axes[r][c].set_position([cur.x0, cur.y0, w, cur.y1 - cur.y0])
out_path = os.path.join(_SCRIPT_DIR, "matsui_2025_all_pitches.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
