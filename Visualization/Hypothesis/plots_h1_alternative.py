import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

#in this version, the y-limits based on the combined range across both conditions
#the original version had separate y-limits for each condition - which made it harder to visually compare the two

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden/Year 2/Thesis Project/2024_data/replication_processing.csv")

sns.set_style("ticks")
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

hv       = fm.FontProperties(family='Helvetica', size=12)
hv_large = fm.FontProperties(family='Helvetica', size=14)

measures = [
    ('fa_rate',            'Lapse rate (%)',          'percent', 5),   
    ('slowest_quintile',   'Slowest 1/5th of trials (%)', 'percent', 10),  
    ('RT_avg',             'Response time (s)',      'sec',     0.5), 
    ('rtcv',               'RT SD / mean RT',        'ratio',   0.2)  
]

colors = {
    1: "#840000",
    0: "#daa800"
}

y_limits = {}
for col, _, scale_type, tick_step in measures:
    y_min_all = np.inf
    y_max_all = -np.inf

    for inst in [0, 1]:
        df_sub = df[df['instructions'] == inst]
        pivot = df_sub.pivot_table(
            index='subject',
            columns=df_sub.groupby('subject').cumcount(),
            values=col
        )
        mean_series = pivot.mean(axis=0)
        sem_series = pivot.sem(axis=0)
        if scale_type == 'percent':
            mean_series = mean_series * 100
            sem_series  = sem_series * 100

        y_min_all = min(y_min_all, np.nanmin(mean_series - sem_series))
        y_max_all = max(y_max_all, np.nanmax(mean_series + sem_series))

    y_min_all = np.floor(y_min_all / tick_step) * tick_step
    y_max_all = np.ceil(y_max_all / tick_step) * tick_step
    y_limits[col] = (y_min_all, y_max_all)

def make_2x2_plot(data, color, filename):

    data = data.copy()
    data['window_index'] = data.groupby('subject').cumcount()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for ax, (col, label, scale_type, tick_step) in zip(axes, measures):

        pivot = data.pivot_table(
            index='subject',
            columns='window_index',
            values=col
        )
        mean_series = pivot.mean(axis=0)
        sem_series  = pivot.sem(axis=0)

        if scale_type == 'percent':
            mean_series = mean_series * 100
            sem_series  = sem_series * 100

        x = np.linspace(0, 1, len(mean_series))
        y = mean_series.values
        sem = sem_series.values

        ax.plot(x, y, color=color, linewidth=2.2)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.2)

        ax.set_xlabel('Time on task', fontproperties=hv_large, labelpad=10)
        ax.set_ylabel(label, fontproperties=hv_large, labelpad=10)

        tick_positions = [0, 1]
        tick_labels    = ['Start', 'End']
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        y_min, y_max = y_limits[col]
        ax.set_ylim(y_min, y_max)
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

for inst in [1, 0]:
    df_sub = df[df['instructions'] == inst]
    color  = colors[inst]

    make_2x2_plot(
        df_sub,
        color,
        f"behavior_2x2_instruction_alternative_{inst}.png"
    )