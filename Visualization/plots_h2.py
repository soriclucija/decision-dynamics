import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\replication_processing.csv")

sns.set_style("ticks")
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

hv       = fm.FontProperties(family='Helvetica', size=12)
hv_large = fm.FontProperties(family='Helvetica', size=14)

def plot_baseline_derivative(df_sub, color, filename):

    df_sub = df_sub.copy()
    df_sub['window_index'] = df_sub.groupby('subject').cumcount()

    fig, axes = plt.subplots(2, 1, figsize=(4, 8))  # 2 rows, 1 column
    measures = [('baseline_z', 'Pupil diameter (z)'), 
                ('derivative', 'Pupil diameter derivative')]

    for ax, (col, label) in zip(axes, measures):
        pivot = df_sub.pivot_table(index='subject', columns='window_index', values=col)
        mean_series = pivot.mean(axis=0)
        sem_series  = pivot.sem(axis=0)

        x = np.linspace(0, 1, len(mean_series))  # normalized time
        y = mean_series.values
        sem = sem_series.values

        ax.plot(x, y, color=color, linewidth=2.5)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.2)

        ax.set_xlabel('Time on task', fontproperties=hv_large, labelpad=10)
        ax.set_ylabel(label, fontproperties=hv_large, labelpad=10)

        tick_positions = [0, 1]
        tick_labels    = ['Start', 'End']

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
               
        if col == 'baseline_z':
            y_min = np.floor(np.nanmin(y - sem) * 2) / 2  # round down to nearest 0.5
            y_max = np.ceil(np.nanmax(y + sem) * 2) / 2   # round up to nearest 0.5
            ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.5))  # step = 0.5

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

colors = {1: '#7eabc2', 0: '#00696e'}

for inst in [1, 0]:
    df_sub = df[df['instructions'] == inst]
    plot_baseline_derivative(
        df_sub,
        color=colors[inst],
        filename=f'pupil_baseline_derivative_instruction_{inst}.png'
    )