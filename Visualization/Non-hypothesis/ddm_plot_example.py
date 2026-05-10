import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

v   = 0.5   
a   = 1.0    
z   = 0.5   
t0  = 0.25   
s   = 0.25    
dt  = 0.001  

#example
rt       = 1.187   # RT (s)
response = "correct"

upper =  a / 2
lower = -a / 2
x0    =  z * a - a / 2

for seed in range(5000):
    np.random.seed(seed)
    x     = x0
    path  = [x]
    times = [0.0]
    while True:
        dx = v * dt + s * np.sqrt(dt) * np.random.randn()
        x += dx
        times.append(times[-1] + dt)
        path.append(x)
        if x >= upper:
            hit = "correct"
            break
        if x <= lower:
            hit = "error"
            break
    if hit == "correct":
        break

decision_time = rt - t0
scale = decision_time / times[-1]
times = [t * scale for t in times]

import matplotlib.font_manager as fm
available = [f.name for f in fm.fontManager.ttflist]
plt.rcParams["font.family"] = "Helvetica" if "Helvetica" in available else "Arial"

upper_col = "#001670"
lower_col = "#daa800"

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


ax.axvspan(0, t0, ymin=0.22, ymax=0.84, color="#f7f7f7f4", zorder=0)
ax.text(t0 / 2, 0.3, "Non-decision\ntime",
        ha="center", va="center", fontsize=12, color="#000000", fontweight="bold")

ax.axhline(upper, color=upper_col, linewidth=2.0, linestyle="--", zorder=2)
ax.axhline(lower, color=lower_col, linewidth=2.0, linestyle="--", zorder=2)

xlim_right = rt + 0.15
ax.annotate("", xy=(xlim_right, 0), xytext=(t0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=1, zorder=1))
ax.text(xlim_right + 0.02, 0, "Time", ha="left", va="center",
        fontsize=12, color="#aaaaaa")
ax.text(xlim_right + 0.03, upper + 0.03, "Right", color=upper_col,
        fontsize=14, fontweight="bold", ha="right", va="bottom")
ax.text(xlim_right + 0.03, lower - 0.03, "Left", color=lower_col,
        fontsize=14, fontweight="bold", ha="right", va="top")

plot_times = [t + t0 for t in times]
points = np.array([plot_times, path]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
colors = [upper_col if (path[i] + path[i+1]) / 2 > 0 else lower_col
          for i in range(len(path) - 1)]

lc = LineCollection(segments, colors=colors, linewidth=1.7, alpha=0.75, zorder=3)
ax.add_collection(lc)
ax.plot([0, t0], [z * a - a / 2, z * a - a / 2],
        color="#000000", linewidth=1.2, linestyle="--", alpha=0.18, zorder=2)

ax.plot(t0, z * a - a / 2, "o", color="#000000", markersize=12, zorder=5)

hit_color = upper_col if response == "correct" else lower_col
ax.plot(rt, path[-1], "o", color=hit_color, markersize=14, zorder=6)

ax.set_xlabel("")
ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax.set_ylabel("")
ax.set_title("")
ax.set_xlim(0, xlim_right + 0.12)
ax.set_ylim(lower - 0.35, upper + 0.25)
ax.set_yticks([lower, 0, upper])
ax.set_yticklabels(["", "Starting\npoint", ""], fontsize=12)
plt.setp(ax.get_yticklabels(), multialignment='center')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", which="both", left=False, bottom=False)
ax.get_legend().set_visible(False) if ax.get_legend() else None

plt.tight_layout()
plt.savefig("ddm_visualization.png", dpi=800, bbox_inches="tight")
plt.show()
print(f"\nResponse : {response}")
print(f"RT       : {rt:.3f} s  (decision: {decision_time:.3f} s + t0: {t0:.3f} s)")