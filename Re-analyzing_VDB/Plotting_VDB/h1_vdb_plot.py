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

measures = [
    ('fa_rate_z',          'Lapse rate (z, %)',               'percent', 25),
    ('slowest_quintile_z', 'Slowest 1/5th of trial (z, %)',  'percent', 25),
    ('RT_avg_z',           'Response time (z, s)',             'sec',     0.25),
    ('rtcv_z',             'RT SD / mean RT (z)',               'ratio',   0.25),
]

COLOR = "#2e2e2e"


def make_2x2_plot(data, filename):

    # Average across blocks per participant × window
    data_avg = (
        data
        .groupby(['participant', 'window'])[
            [col for col, *_ in measures]
        ]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for ax, (col, label, scale_type, tick_step) in zip(axes, measures):

        pivot = data_avg.pivot_table(
            index='participant',
            columns='window',
            values=col
        )

        mean_series = pivot.mean(axis=0)
        sem_series  = pivot.sem(axis=0)

        if scale_type == 'percent':
            mean_series = mean_series * 100
            sem_series  = sem_series  * 100

        x   = np.linspace(0, 1, len(mean_series))
        y   = mean_series.values
        sem = sem_series.values

        ax.plot(x, y, color=COLOR, linewidth=2.2)
        ax.fill_between(x, y - sem, y + sem, color=COLOR, alpha=0.2)

        ax.set_xlabel('Time on task',  fontproperties=hv_large, labelpad=4)
        ax.set_ylabel(label,           fontproperties=hv_large, labelpad=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Start', 'End'])

        y_min = np.floor((np.nanmin(y - sem)) / tick_step) * tick_step
        y_max = np.ceil( (np.nanmax(y + sem)) / tick_step) * tick_step
        ax.set_yticks(np.arange(y_min, y_max + 1e-6, tick_step))

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


make_2x2_plot(df, "behavior_2x2_VDB.png")
