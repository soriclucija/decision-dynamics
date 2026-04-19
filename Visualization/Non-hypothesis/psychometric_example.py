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


def plot_min_only(df, ax, xlims):
    subjects = df['subject'].unique()
    all_curves    = []
    fitted_params = []

    for subj in subjects:
        pars = fit_one_subject(df[df['subject'] == subj])
        if pars is not None:
            all_curves.append(erf_psycho_2gammas(pars, xlims))
            fitted_params.append(pars)

    if len(all_curves) == 0:
        print("WARNING: No valid fits for Min condition")
        return None

    all_curves = np.array(all_curves)
    mean_curve = all_curves.mean(axis=0)

    ax.plot(xlims, mean_curve, color='black', linewidth=2.5)

    return fitted_params


def main():
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns.tolist())
    print("Instructions values:", df['instructions'].unique())
    print("Choice values:", df['choice'].unique())
    print(f"N trials: {len(df)},  N subjects: {df['subject'].nunique()}")

    df_min = df[df['instructions'] == 0]

    xmax  = np.max(np.abs(df_min['signed_contrast'].dropna()))
    xlims = np.linspace(-xmax, xmax, 200)

    sns.set_style("ticks")
    mpl.rcParams['font.family']     = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']

    hv_large = fm.FontProperties(family='Helvetica', size=16)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    plot_min_only(df_min, ax, xlims)

    ax.axvline(x=0, color='gray', linewidth=1, linestyle='--', zorder=1)


    xticks = [-0.2, 0, 0.2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], rotation=70, ha='center',
                        fontsize=12, color='black')
    ax.tick_params(axis='x', pad=8)

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0.00', '0.50', '1.00'],
                        fontsize=12, color='black')
    ax.tick_params(axis='y', colors='black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    ax.set_xlabel('Contrast (level)', color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.set_ylabel('Proportion (choice = 1)', color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.set_ylim(0, 1)

    sns.despine(trim=True)

    for artist in ax.lines + ax.collections:
        artist.set_clip_on(False)

    plt.tight_layout()
    plt.savefig('psychometric_example.png', dpi=600, bbox_inches='tight')
    plt.show()

    print("\nFigure saved as psychometric_example.png")


if __name__ == '__main__':
    main()