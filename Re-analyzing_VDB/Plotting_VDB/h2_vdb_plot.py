import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

df = pd.read_csv(r"C:\Users\lucij\Documents\combined_behavior_data_VDB.csv")

sns.set_style("ticks")
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

hv       = fm.FontProperties(family='Helvetica', size=12)
hv_large = fm.FontProperties(family='Helvetica', size=14)

COLOR = "#2e2e2e"

pupil_measures = [
    ('baseline_z',   'Pupil diameter (z)'),
    ('derivative_z', 'Pupil diameter derivative (z)'),
]

def plot_baseline_derivative(data, filename):

    # Average across blocks per participant × window
    data_avg = (
        data
        .groupby(['participant', 'window'])[
            [col for col, _ in pupil_measures]
        ]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(4, 8))

    for ax, (col, label) in zip(axes, pupil_measures):

        pivot = data_avg.pivot_table(
            index='participant',
            columns='window',
            values=col
        )

        mean_series = pivot.mean(axis=0)
        sem_series  = pivot.sem(axis=0)

        x   = np.linspace(0, 1, len(mean_series))
        y   = mean_series.values
        sem = sem_series.values

        ax.plot(x, y, color=COLOR, linewidth=2.5)
        ax.fill_between(x, y - sem, y + sem, color=COLOR, alpha=0.2)

        ax.set_xlabel('Time on task', fontproperties=hv_large, labelpad=4)
        ax.set_ylabel(label,          fontproperties=hv_large, labelpad=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Start', 'End'])

        y_min = np.floor(np.nanmin(y - sem) * 2) / 2
        y_max = np.ceil( np.nanmax(y + sem) * 2) / 2
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.5))

        for tick in ax.get_xticklabels():
            tick.set_fontproperties(hv)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(hv)

        ax.tick_params(axis='both', which='both',
                       length=4, width=1,
                       direction='out', pad=6,
                       colors='black', labelsize=11)

        sns.despine(ax=ax, trim=False)
        ax.spines['bottom'].set_position(('outward', 6))
        ax.spines['left'].set_position(('outward', 6))

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()


plot_baseline_derivative(df, filename='pupil_baseline_derivative_vdb.png')