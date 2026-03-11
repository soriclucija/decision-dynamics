import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import erf

DATA_PATH = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min",     1: "Full"}

XRANGE   = np.linspace(-1, 1, 200) 

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
                (x.min(), x.max()),  # mu
                (1e-4, 1.0),         # sigma
                (0.0, 0.4),          # gamma
                (0.0, 0.4),          # lambda
            ],
            method='L-BFGS-B'
        )
        if result.success or result.fun < 1e6:
            return result.x
    except Exception:
        pass
    return None


def plot_condition(df, color, label, ax, xlims):
    subjects = df['subject'].unique()
    all_curves = []
    fitted_params = []

    for subj in subjects:
        subj_df = df[df['subject'] == subj]
        params = fit_one_subject(subj_df)
        if params is not None:
            y = erf_psycho_2gammas(params, xlims)
            all_curves.append(y)
            fitted_params.append(params)

    if len(all_curves) == 0:
        print(f"  WARNING: No valid fits for condition {label}")
        return None

    all_curves = np.array(all_curves)  # shape: (n_subjects, n_xpoints)

    mean_curve = all_curves.mean(axis=0)
    sem_curve  = all_curves.std(axis=0) / np.sqrt(len(all_curves))

    ax.plot(xlims, mean_curve, color=color, linewidth=2.5, zorder=10, label=label)

    ax.fill_between(xlims,
                    mean_curve - sem_curve,
                    mean_curve + sem_curve,
                    color=color, alpha=0.15, zorder=5)
    
    df = df.dropna(subset=['signed_contrast', 'choice']).copy()
    choice_vals = sorted(df['choice'].unique())
    df['choice_binary'] = (df['choice'] == choice_vals[-1]).astype(int)

    subj_props = (df.groupby(['subject', 'signed_contrast'])['choice_binary']
                    .mean()
                    .reset_index())
    mean_props = subj_props.groupby('signed_contrast')['choice_binary'].agg(['mean', 'sem']).reset_index()

# Draw markers first (lower zorder)
    ax.scatter(
        mean_props['signed_contrast'],
        mean_props['mean'],
        color=color,
        s=12**2,
        zorder=20,
        edgecolors='white',
        linewidths=2,
    )

# Draw error bars on top (higher zorder)
    ax.errorbar(
        mean_props['signed_contrast'],
        mean_props['mean'],
        yerr=mean_props['sem'],
        fmt='none',
        color=color,
        linewidth=3,
        zorder=21,
    )
    

    params_arr = np.array(fitted_params)
    print(f"\n{label} ({len(fitted_params)} subjects fitted):")
    print(f"  μ = {params_arr[:,0].mean():.3f} ± {params_arr[:,0].std():.3f}")
    print(f"  σ = {params_arr[:,1].mean():.3f} ± {params_arr[:,1].std():.3f}")
    print(f"  γ = {params_arr[:,2].mean():.3f} ± {params_arr[:,2].std():.3f}")
    print(f"  λ = {params_arr[:,3].mean():.3f} ± {params_arr[:,3].std():.3f}")

    return fitted_params

def main():
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns.tolist())
    print("Instructions values:", df['instructions'].unique())
    print("Choice values:", df['choice'].unique())
    print(f"N trials: {len(df)},  N subjects: {df['subject'].nunique()}")

    xmax = np.max(np.abs(df['signed_contrast'].dropna()))
    xlims = np.linspace(-xmax, xmax, 200)

    plt.rcParams['font.family'] = 'Helvetica'
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for condition in [0, 1]:
        subset = df[df['instructions'] == condition]
        plot_condition(subset, COLORS[condition], LABELS[condition], ax, xlims)

    xticks = [-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], rotation=70, ha='center', fontsize=12, color='black')
    ax.tick_params(axis='x', pad=8)

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=12, color='black')
    ax.tick_params(axis='y', colors='black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    ax.set_xlabel('Signed contrast', fontsize=16, color='black', labelpad=14)
    ax.set_ylabel('Proportion rightward choice', fontsize=16, color='black', labelpad=14)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v)) if v == int(v) else f'{v:.1f}'))

    legend = ax.get_legend()
    if legend:
        legend.remove()

    ax.text(0.05, 0.97, '', transform=ax.transAxes)  
    ax.text(0.05, 0.97,
            r'$\bf{Min.}$' + r' ($\it{N}$ = 37)',
            transform=ax.transAxes, fontsize=16,
            color=COLORS[0], va='top', ha='left')
    ax.text(0.05, 0.91,
            r'$\bf{Full}$' + r' ($\it{N}$ = 35)',
            transform=ax.transAxes, fontsize=16,
            color=COLORS[1], va='top', ha='left')

    sns.despine(trim=True)

    for artist in ax.lines + ax.collections:
        artist.set_clip_on(False)

    plt.tight_layout()
    plt.savefig('psychometric_choice.png', dpi=600, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as psychometric_choice.png")


if __name__ == '__main__':
    main()