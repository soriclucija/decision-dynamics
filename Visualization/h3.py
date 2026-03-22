import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\replication_processing.csv")

sns.set_style("ticks")
fm.fontManager.addfont(r"C:\Users\lucij\AppData\Local\Microsoft\Windows\Fonts\Helvetica.ttf")
fm.fontManager.addfont(r"C:\Users\lucij\AppData\Local\Microsoft\Windows\Fonts\Helvetica-Bold_0.ttf")
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

colors = {
    1: "#840000",
    0: "#daa800"
}

labels = {
    1: "With instructions",
    0: "No instructions"
}

pupil_cols      = ['baseline_z', 'derivative_z']
behavior_cols   = ['rtcv_z', 'slowest_quintile_z', 'RT_avg_z', 'fa_rate_z']
behavior_labels = ['RTCV', 'Slowest quintile', 'Mean RT', 'Lapse rate']


def get_coefficients(df_inst):
    results = {}

    for pupil in pupil_cols:
        for behav in behavior_cols:
            lin_coefs, quad_coefs = [], []
            for _, grp in df_inst.groupby('subject'):
                y    = grp[behav].values
                x    = grp[pupil].values
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() < 4:
                    continue
                X     = np.column_stack([np.ones(mask.sum()), x[mask], x[mask] ** 2])
                coefs, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
                lin_coefs.append(coefs[1])
                quad_coefs.append(coefs[2])

            lin_coefs  = np.array(lin_coefs)
            quad_coefs = np.array(quad_coefs)

            _, p_lin  = stats.ttest_1samp(lin_coefs,  popmean=0)
            _, p_quad = stats.ttest_1samp(quad_coefs, popmean=0)

            results[(pupil, behav, 'linear',    'uncontrolled')] = {
                'mean': lin_coefs.mean(), 'sem': stats.sem(lin_coefs), 'p': p_lin
            }
            results[(pupil, behav, 'quadratic', 'uncontrolled')] = {
                'mean': quad_coefs.mean(), 'sem': stats.sem(quad_coefs), 'p': p_quad
            }

    for pupil in pupil_cols:
        for behav in behavior_cols:
            lin_coefs, quad_coefs = [], []
            for _, grp in df_inst.groupby('subject'):
                y    = grp[behav].values
                x    = grp[pupil].values
                win  = grp['window'].values
                mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(win)
                if mask.sum() < 5:
                    continue
                X     = np.column_stack([np.ones(mask.sum()), win[mask], x[mask], x[mask] ** 2])
                coefs, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
                lin_coefs.append(coefs[2])
                quad_coefs.append(coefs[3])

            lin_coefs  = np.array(lin_coefs)
            quad_coefs = np.array(quad_coefs)

            _, p_lin  = stats.ttest_1samp(lin_coefs,  popmean=0)
            _, p_quad = stats.ttest_1samp(quad_coefs, popmean=0)

            results[(pupil, behav, 'linear',    'controlled')] = {
                'mean': lin_coefs.mean(), 'sem': stats.sem(lin_coefs), 'p': p_lin
            }
            results[(pupil, behav, 'quadratic', 'controlled')] = {
                'mean': quad_coefs.mean(), 'sem': stats.sem(quad_coefs), 'p': p_quad
            }

    return results


all_results = {}
for inst in [0, 1]:
    all_results[inst] = get_coefficients(df[df['instructions'] == inst])


plot_specs = [
    ('baseline_z',   'linear',    'uncontrolled', 'coef_baseline_linear.png'),
    ('baseline_z',   'quadratic', 'uncontrolled', 'coef_baseline_quadratic.png'),
    ('baseline_z',   'linear',    'controlled',   'coef_baseline_linear_controlled.png'),
    ('baseline_z',   'quadratic', 'controlled',   'coef_baseline_quadratic_controlled.png'),
    ('derivative_z', 'linear',    'uncontrolled', 'coef_derivative_linear.png'),
    ('derivative_z', 'linear',    'controlled',   'coef_derivative_linear_controlled.png'),
]


def make_coef_plot(pupil, coef_type, controlled, filename):

    fig, ax = plt.subplots(figsize=(5, 5))

    marker      = 's' if coef_type == 'quadratic' else 'o'
    msize       = 12
    mew         = 1.5
    y_positions = np.arange(len(behavior_cols), dtype=float)
    offsets     = {1: 0.18, 0: -0.18}

    ax.axvline(x=0, color='#fdeabe', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

    for inst in [1, 0]:
        color = colors[inst]

        for i, behav in enumerate(behavior_cols):
            vals = []

            for _, grp in df[df['instructions'] == inst].groupby('subject'):
                y = grp[behav].values
                x = grp[pupil].values

                if controlled == 'controlled':
                    win  = grp['window'].values
                    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(win)
                    if mask.sum() < 5:
                        continue
                    X = np.column_stack([
                        np.ones(mask.sum()), win[mask], x[mask], x[mask] ** 2
                    ])
                    coefs, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
                    lin_idx, quad_idx = 2, 3  

                else:  # uncontrolled
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    if mask.sum() < 4:
                        continue
                    X = np.column_stack([
                        np.ones(mask.sum()), x[mask], x[mask] ** 2
                    ])
                    coefs, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
                    lin_idx, quad_idx = 1, 2

                if coef_type == 'linear':
                    vals.append(coefs[lin_idx])
                else:
                    vals.append(coefs[quad_idx])

            vals = np.array(vals)
            mean = vals.mean()
            sem  = stats.sem(vals)
            yp   = y_positions[i] + offsets[inst]

            ax.errorbar(
                mean, yp,
                xerr=sem,
                fmt=marker,
                color=color,
                markersize=msize,
                markeredgecolor='white',
                markeredgewidth=mew,
                elinewidth=1.5,
                zorder=4
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(behavior_labels)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Regression coefficient (β)', fontsize=14, labelpad=5)
    ax.tick_params(axis='x', length=4, width=1, direction='out', pad=8,
                   colors='black', labelsize=10)

    x_min, x_max = -0.2, 0.2
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max + 1e-9, 0.1))

    sns.despine(ax=ax, trim=False)
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['left'].set_position(('outward', 6))

    plt.tight_layout(pad=1.2)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


for pupil, coef_type, controlled, filename in plot_specs:
    make_coef_plot(pupil, coef_type, controlled, filename)