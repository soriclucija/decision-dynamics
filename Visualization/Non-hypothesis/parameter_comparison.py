import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import ttest_ind

sys.path.append(r"C:\Users\lucij\Documents\GitHub\Thesis_Project\psychofit")
from psychofit import mle_fit_psycho, erf_psycho_2gammas

DATA_PATH = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min.",    1: "Full"}


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

    data = grouped[['signed_contrast', 'ntrials', 'fraction']].transpose().values

    try:
        pars, L = mle_fit_psycho(
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


def plot_panel(ax, params_df, col, ylabel):
    for cond, x_pos in [(0, 0), (1, 1)]:
        vals  = params_df[params_df['condition'] == cond][col].values
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
    ax.tick_params(axis='y', labelsize=14, colors='black')
    ax.set_xlim(-0.9, 1.9)

    for x_pos, cond in [(0, 0), (1, 1)]:
        ax.text(
            x_pos,
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12,
            LABELS[cond],
            ha='center', va='top',
            fontsize=16, fontweight='bold', color=COLORS[cond],
            transform=ax.transData,
        )

    ax.set_ylabel(ylabel, fontsize=16, color='black', labelpad=14)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['bottom'].set_bounds(-0.2, 1.2)


def run_stats(params_df):
    print("\n─── Statistical Tests (Welch t-test) ───")
    results = {}

    for col, name in [
        ('sigma', 'Threshold (σ)'),
        ('lapse', 'Lapse rate (λ)'),
        ('mu',    'Bias'),
    ]:
        vals0 = params_df[params_df['condition'] == 0][col].values
        vals1 = params_df[params_df['condition'] == 1][col].values

        stat, p = ttest_ind(vals0, vals1, equal_var=False)
        df_deg  = len(vals0) + len(vals1) - 2

        if p < 0.001:
            p_str, stars = "< .001", "***"
        elif p < 0.01:
            p_str, stars = "< .01",  "**"
        elif p < 0.05:
            p_str, stars = "< .05",  "*"
        else:
            p_str, stars = f"= {p:.3f}", "ns"

        print(f"  {name}: t({df_deg}) = {stat:.2f}, p {p_str}  {stars}")
        results[col] = p

    return results


def main():
    df = pd.read_csv(DATA_PATH)

    records = []
    for condition in [0, 1]:
        subset = df[df['instructions'] == condition]
        for subj in subset['subject'].unique():
            pars = fit_one_subject(subset[subset['subject'] == subj])
            if pars is not None:
                mu, sigma, gamma, lam = pars
                records.append({
                    'condition': condition,
                    'subject':   subj,
                    'mu':        mu,
                    'sigma':     sigma,
                    'lapse':     lam,
                })

    params_df = pd.DataFrame(records)
    print(params_df.groupby('condition')[['sigma', 'lapse', 'mu']].describe())

    p_vals = run_stats(params_df)

    sns.set_style("ticks")
    mpl.rcParams['font.family']     = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    panels = [
        (axes[0], 'sigma', 'Threshold'),
        (axes[1], 'lapse', 'Lapse rate (proportion)'),
    ]

    for ax, col, ylabel in panels:
        plot_panel(ax, params_df, col, ylabel)

        if col == 'sigma':
            ax.set_yticks([0, 0.4, 0.8])
        if col == 'lapse':
            ax.set_yticks([0.0, 0.15, 0.30])


    sns.despine(trim=False)
    plt.tight_layout(pad=3.0)
    plt.savefig('parameter_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as parameter_comparison.png")


if __name__ == '__main__':
    main()