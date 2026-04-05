import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

sns.set_style("ticks")

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['axes.formatter.use_mathtext'] = False

helvetica = fm.FontProperties(family='Helvetica')

df = pd.read_csv(
    r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"
)

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min.",     1: "Full"}

# participant-condition time out rate
timeout_rates = (
    df.groupby(['subject', 'instructions'])
      .apply(lambda g: g['timeout'].sum() / len(g), include_groups=False)
      .reset_index(name='timeout_rate')
)

fig, ax = plt.subplots(figsize=(5.5, 5.5))

for cond, x_pos in [(0, 0), (1, 1)]:
    vals = timeout_rates[timeout_rates['instructions'] == cond]['timeout_rate'].values
    color = COLORS[cond]

    ax.boxplot(
        vals,
        positions=[x_pos],
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.25, linewidth=1.2, edgecolor=color),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
    )

    rng = np.random.default_rng(seed=42)
    jitter = rng.uniform(-0.09, 0.09, size=len(vals))
    ax.scatter(
        x_pos + jitter,
        vals,
        color=color,
        s=35,
        alpha=0.65,
        zorder=5,
        edgecolors='white',
        linewidths=0.4,
    )

ax.set_xticks([0, 1])
ax.set_xticklabels([])
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', labelsize=14, colors='black')
ax.set_xlim(-0.9, 1.9)

# fonts
for label in ax.get_yticklabels():
    label.set_fontproperties(helvetica)
    label.set_fontsize(14)

ax.set_ylabel('Timeout rate (%)', fontsize=16, color='black', labelpad=14,
              fontproperties=helvetica)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))

for x_pos, cond in [(0, 0), (1, 1)]:
    ax.text(
        x_pos,
        ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12,
        LABELS[cond],
        ha='center', va='top',
        color=COLORS[cond],
        fontproperties=fm.FontProperties(family='Helvetica', size=16, weight='bold'),
        transform=ax.transData,
    )

for spine in ax.spines.values():
    spine.set_edgecolor('black')
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['bottom'].set_bounds(-0.2, 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('timeout_rate_by_condition.png', dpi=600, bbox_inches='tight')
plt.show()