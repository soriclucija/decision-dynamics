import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns

DATA_PATH = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min.",     1: "Full"}

def main():
    df = pd.read_csv(DATA_PATH)

    sns.set_style("ticks")
    mpl.rcParams['font.family'] = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']

    hv       = fm.FontProperties(family='Helvetica', size=12)
    hv_large = fm.FontProperties(family='Helvetica', size=16)

    df_rt = df[(df['timeout'] == 0) & df['response_time'].notna()]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for condition in [1, 0]:  
        color  = COLORS[condition]
        subset = df_rt[df_rt['instructions'] == condition]

        subj_medians = (
            subset.groupby(['subject', 'signed_contrast'])['response_time']
                  .median()
                  .reset_index()
        )
        stats = (
            subj_medians.groupby('signed_contrast')['response_time']
                        .agg(['mean', 'sem'])
                        .reset_index()
        )
        ax.plot(
            stats['signed_contrast'],
            stats['mean'],
            color=color,
            linewidth=2.5,
            zorder=10,
        )
        ax.scatter(
            stats['signed_contrast'],
            stats['mean'],
            color=color,
            s=12**2,
            zorder=20,
            edgecolors='white',
            linewidths=2,
            alpha=0.7,
        )
        ax.errorbar(
            stats['signed_contrast'],
            stats['mean'],
            yerr=stats['sem'],
            fmt='none',
            color=color,
            linewidth=3,
            zorder=21,
            alpha=1,
        )

    xticks = [-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], rotation=70, ha='center', color='black')
    ax.tick_params(axis='x', pad=8)
    for label in ax.get_xticklabels():
        label.set_fontproperties(hv)

    ax.tick_params(axis='y', colors='black')
    for label in ax.get_yticklabels():
        label.set_fontproperties(hv)

    ax.set_xlabel('Signed contrast', color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.set_ylabel('Median RT (s)', color='black', labelpad=14,
                  fontproperties=hv_large)

    ax.set_ylim(0, 1.4)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    sns.despine(trim=True)

    for artist in ax.lines + ax.collections:
        artist.set_clip_on(False)

    plt.tight_layout()
    plt.savefig('chronometric_curve.png', dpi=600, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as chronometric_curve.png")


if __name__ == '__main__':
    main()