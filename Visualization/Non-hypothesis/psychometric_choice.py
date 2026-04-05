import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

sys.path.append(r"C:\Users\lucij\Documents\GitHub\Thesis_Project\psychofit")
from psychofit import mle_fit_psycho, erf_psycho_2gammas

DATA_PATH = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_dataset.csv"

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min",     1: "Full"}


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
            parstart=np.array([0, 0.2, 0.05, 0.05]),
            parmin=np.array([x_min, 1e-4, 0.0, 0.0]),
            parmax=np.array([x_max, 1.0,  0.4, 0.4]),
            nfits=10
        )
        return pars
    except Exception:
        return None


def plot_condition(df, color, label, ax, xlims):
    subjects = df['subject'].unique()
    all_curves    = []
    fitted_params = []

    for subj in subjects:
        pars = fit_one_subject(df[df['subject'] == subj])
        if pars is not None:
            all_curves.append(erf_psycho_2gammas(pars, xlims))
            fitted_params.append(pars)

    if len(all_curves) == 0:
        print(f"  WARNING: No valid fits for condition {label}")
        return None

    all_curves = np.array(all_curves)          
    mean_curve = all_curves.mean(axis=0)
    sem_curve  = all_curves.std(axis=0) / np.sqrt(len(all_curves))

    ax.plot(xlims, mean_curve, color=color, linewidth=2.5, zorder=10, label=label)
    ax.fill_between(
        xlims,
        mean_curve - sem_curve,
        mean_curve + sem_curve,
        color=color, alpha=0.15, zorder=5
    )

    df = df.dropna(subset=['signed_contrast', 'choice']).copy()
    choice_vals = sorted(df['choice'].unique())
    df['choice_binary'] = (df['choice'] == choice_vals[-1]).astype(int)

    subj_props = (df.groupby(['subject', 'signed_contrast'])['choice_binary']
                    .mean()
                    .reset_index())
    mean_props = (subj_props.groupby('signed_contrast')['choice_binary']
                             .agg(['mean', 'sem'])
                             .reset_index())

    ax.scatter(
        mean_props['signed_contrast'], mean_props['mean'],
        color=color, s=12**2, zorder=20,
        edgecolors='white', linewidths=2,
    )
    ax.errorbar(
        mean_props['signed_contrast'], mean_props['mean'],
        yerr=mean_props['sem'],
        fmt='none', color=color, linewidth=3, zorder=21,
    )

    params_arr = np.array(fitted_params)
    print(f"\n{label} ({len(fitted_params)} subjects fitted):")
    print(f"  μ (bias)  = {params_arr[:,0].mean():.3f} ± {params_arr[:,0].std():.3f}")
    print(f"  σ (slope) = {params_arr[:,1].mean():.3f} ± {params_arr[:,1].std():.3f}")
    print(f"  γ (lapse low)  = {params_arr[:,2].mean():.3f} ± {params_arr[:,2].std():.3f}")
    print(f"  λ (lapse high) = {params_arr[:,3].mean():.3f} ± {params_arr[:,3].std():.3f}")

    return fitted_params


def main():
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns.tolist())
    print("Instructions values:", df['instructions'].unique())
    print("Choice values:", df['choice'].unique())
    print(f"N trials: {len(df)},  N subjects: {df['subject'].nunique()}")

    xmax  = np.max(np.abs(df['signed_contrast'].dropna()))
    xlims = np.linspace(-xmax, xmax, 200)

    sns.set_style("ticks")
    mpl.rcParams['font.family']     = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']

    hv_large = fm.FontProperties(family='Helvetica', size=16)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for condition in [0, 1]:
        subset = df[df['instructions'] == condition]
        plot_condition(subset, COLORS[condition], LABELS[condition], ax, xlims)

    xticks = [-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], rotation=70, ha='center',
                        fontsize=12, color='black')
    ax.tick_params(axis='x', pad=8)

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'],
                        fontsize=12, color='black')
    ax.tick_params(axis='y', colors='black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    ax.set_xlabel('Signed contrast',              color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.set_ylabel('Rightward choice (proportion)', color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.set_ylim(0, 1)

    legend = ax.get_legend()
    if legend:
        legend.remove()

    sns.despine(trim=True)

    for artist in ax.lines + ax.collections:
        artist.set_clip_on(False)

    plt.tight_layout()
    plt.savefig('psychometric_choice.png', dpi=600, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as psychometric_choice.png")


if __name__ == '__main__':
    main()