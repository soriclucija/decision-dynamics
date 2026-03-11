import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import mannwhitneyu

DATA_PATH = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min.",    1: "Full"}

def erf_psycho_2gammas(params, x):
    mu, sigma, gamma, lam = params
    return gamma + (1 - gamma - lam) * (0.5 + 0.5 * erf((x - mu) / (np.sqrt(2) * sigma)))


def neg_log_likelihood(params, x, n, k):
    mu, sigma, gamma, lam = params
    if sigma <= 0 or gamma < 0 or lam < 0 or gamma > 1 or lam > 1:
        return np.inf
    p = erf_psycho_2gammas(params, x)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))


def fit_one_subject(df):
    df = df.dropna(subset=['signed_contrast', 'choice'])
    if len(df) < 10:
        return None

    choice_vals = sorted(df['choice'].unique())
    df = df.copy()
    df['choice_binary'] = (df['choice'] == choice_vals[-1]).astype(int)

    grouped = df.groupby('signed_contrast').agg(
        n=('choice_binary', 'count'),
        k=('choice_binary', 'sum')
    ).reset_index()

    if len(grouped) < 3:
        return None

    x = grouped['signed_contrast'].values
    n = grouped['n'].values
    k = grouped['k'].values

    try:
        result = minimize(
            neg_log_likelihood,
            x0=[0, 0.2, 0.05, 0.05],
            args=(x, n, k),
            bounds=[
                (x.min(), x.max()),
                (1e-4, 1.0),
                (0.0, 0.4),
                (0.0, 0.4),
            ],
            method='L-BFGS-B'
        )
        if result.success or result.fun < 1e6:
            return result.x
    except Exception:
        pass
    return None


def plot_panel(ax, params_df, col, ylabel):
    for cond, x_pos in [(0, 0), (1, 1)]:
        vals = params_df[params_df['condition'] == cond][col].values
        color = COLORS[cond]

        # boxplot
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

        # jittered dots
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
    ax.tick_params(axis='x', length=0, pad=14)
    ax.tick_params(axis='y', labelsize=11, colors='black')
    ax.set_xlim(-0.9, 1.9)

    for x_pos, cond in [(0, 0), (1, 1)]:
        ax.text(x_pos, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12,
                LABELS[cond],
                ha='center', va='top',
                fontsize=14, fontweight='bold', color=COLORS[cond],
                transform=ax.transData)

    ax.set_ylabel(ylabel, fontsize=14, color='black', labelpad=14)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['bottom'].set_bounds(-0.2, 1.2)

def run_stats(params_df):
    print("\n─── Statistical Tests (Mann-Whitney U) ───")
    for col, name in [('sigma', 'Threshold'), ('lapse', 'Lapse rate')]:
        vals0 = params_df[params_df['condition'] == 0][col].values
        vals1 = params_df[params_df['condition'] == 1][col].values
        stat, p = mannwhitneyu(vals0, vals1, alternative='two-sided')

        if p < 0.001:
            p_str = "< .001"
            stars = "***"
        elif p < 0.01:
            p_str = "< .01"
            stars = "**"
        elif p < 0.05:
            p_str = "< .05"
            stars = "*"
        else:
            p_str = f"= {p:.3f}"
            stars = "ns"

        print(f"  {name}: U = {stat:.0f}, p {p_str}  {stars}")

def main():
    df = pd.read_csv(DATA_PATH)
    plt.rcParams['font.family'] = 'Helvetica'

    records = []
    for condition in [0, 1]:
        subset = df[df['instructions'] == condition]
        for subj in subset['subject'].unique():
            params = fit_one_subject(subset[subset['subject'] == subj])
            if params is not None:
                records.append({
                    'condition': condition,
                    'subject':   subj,
                    'mu':        params[0],
                    'sigma':     params[1],
                    'lapse':     (params[2] + params[3]) / 2,  # average of gamma and lambda
                })

    params_df = pd.DataFrame(records)
    print(params_df.groupby('condition')[['sigma', 'lapse']].describe())

    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    panels = [
        (axes[0], 'sigma', 'Threshold (proportion)'),
        (axes[1], 'lapse', 'Lapse rate (proportion)'),
    ]

    for ax, col, ylabel in panels:
        plot_panel(ax, params_df, col, ylabel)

        if col == "sigma":  # threshold panel
            ax.set_yticks([0, 0.4, 0.8])

        if col == "lapse":  # lapse panel
            ax.set_yticks([0.0, 0.15, 0.30])

    sns.despine(trim=False)
    plt.tight_layout(pad=3.0)
    plt.savefig('parameter_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as parameter_comparison.png")
    run_stats(params_df)


if __name__ == '__main__':
    main()