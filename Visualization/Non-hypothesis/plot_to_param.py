import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

sys.path.append(r"C:\Users\lucij\Documents\GitHub\Thesis_Project\psychofit")
from psychofit import mle_fit_psycho

DATA_PATH = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min.",    1: "Full"}

helvetica      = fm.FontProperties(family='Helvetica')
helvetica_bold = fm.FontProperties(family='Helvetica', size=20, weight='bold')


def fit_one_subject(df):
    df = df.dropna(subset=['signed_contrast', 'choice'])
    if len(df) < 10:
        return None

    choice_vals = sorted(df['choice'].unique())
    df = df.copy()
    df['choice_binary'] = (df['choice'] == choice_vals[-1]).astype(int)

    grouped = df.groupby('signed_contrast').agg(
        ntrials=('choice_binary', 'count'),
        fraction=('choice_binary', 'mean')
    ).reset_index()

    if len(grouped) < 3:
        return None

    x_min = grouped['signed_contrast'].min()
    x_max = grouped['signed_contrast'].max()
    data  = grouped[['signed_contrast', 'ntrials', 'fraction']].transpose().values

    try:
        pars, _ = mle_fit_psycho(
            data,
            P_model='erf_psycho_2gammas',
            parstart=np.array([0,     0.2,  0.05, 0.05]),
            parmin=  np.array([x_min, 1e-4, 0.0,  0.0 ]),
            parmax=  np.array([x_max, 1.0,  0.4,  0.4 ]),
            nfits=10
        )
        return pars   # [mu, sigma, gamma, lam]
    except Exception:
        return None


def plot_panel(ax, vals_by_cond, ylabel, yticks=None, pct_fmt=False):
    for cond, x_pos in [(0, 0), (1, 0.8)]:
        vals  = vals_by_cond[cond]
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

        rng    = np.random.default_rng(seed=42)
        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(
            x_pos + jitter, vals,
            color=color, s=35, alpha=0.65, zorder=5,
            edgecolors='white', linewidths=0.4,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0, pad=14)
    ax.tick_params(axis='y', labelsize=18, colors='black')
    ax.set_xlim(-0.5, 1.7)

    for label in ax.get_yticklabels():
        label.set_fontproperties(helvetica)
        label.set_fontsize(14)

    if yticks is not None:
        ax.set_yticks(yticks)

    if pct_fmt:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))

    for x_pos, cond in [(0, 0), (0.8, 1)]:
        ax.text(
            x_pos,
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12,
            LABELS[cond],
            ha='center', va='top',
            fontproperties=helvetica_bold,
            color=COLORS[cond],
            transform=ax.transData,
        )

    ax.set_ylabel(ylabel, fontsize=16, color='black', labelpad=14,
                  fontproperties=helvetica)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['bottom'].set_bounds(-0.2, 1)  



def main():
    df = pd.read_csv(DATA_PATH)

    records = []
    for condition in [0, 1]:
        subset = df[df['instructions'] == condition]
        for subj in subset['subject'].unique():
            pars = fit_one_subject(subset[subset['subject'] == subj])
            if pars is not None:
                mu, sigma, gamma, lam = pars
                records.append({'condition': condition, 'subject': subj,
                                'sigma': sigma, 'lapse': lam, 'gamma': gamma})

    params_df = pd.DataFrame(records)

    timeout_rates = (
        df.groupby(['subject', 'instructions'])
          .apply(lambda g: g['timeout'].sum() / len(g), include_groups=False)
          .reset_index(name='timeout_rate')
    ).rename(columns={'instructions': 'condition'})

    sns.set_style("ticks")
    mpl.rcParams['font.family']                 = 'Helvetica'
    mpl.rcParams['font.sans-serif']             = ['Helvetica']
    mpl.rcParams['axes.formatter.use_mathtext'] = False

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    plot_panel(
        axes[0],
        vals_by_cond={c: params_df[params_df['condition'] == c]['gamma'].values for c in [0, 1]},
        ylabel='Lapse rate (left)',
        yticks=[0.0, 0.15, 0.30],
    )

    plot_panel(
        axes[1],
        vals_by_cond={c: params_df[params_df['condition'] == c]['lapse'].values for c in [0, 1]},
        ylabel='Lapse rate (right)',
        yticks=[0.0, 0.15, 0.30],
    )

    plot_panel(
        axes[2],
        vals_by_cond={c: timeout_rates[timeout_rates['condition'] == c]['timeout_rate'].values for c in [0, 1]},
        ylabel='Timeout rate (%)',
        pct_fmt=True,
    )

    plt.tight_layout(pad=0.5)
    plt.savefig('combined_parameter_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()
    print("Figure saved as combined_parameter_comparison.png")


if __name__ == '__main__':
    main()