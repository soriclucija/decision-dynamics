import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns

DATA_PATH = r"C:\Users\lucij\Documents\combined_behavior_data_VDB.csv"

BEHAVIOR_COLS = ['fa_rate_z', 'slowest_quintile_z', 'RT_avg_z', 'rtcv_z']
COLORS = {
    'fa_rate_z':          '#00373d',
    'slowest_quintile_z': '#59acb1',
    'RT_avg_z':           '#d077ee',
    'rtcv_z':             '#4b0088',
}


def get_quadratic_uncontrolled_fits(df, pupil):
    subject_fits = {b: [] for b in BEHAVIOR_COLS}
    for subj, grp_subj in df.groupby('participant'):
        for behav in BEHAVIOR_COLS:
            block_coefs = []
            for block, grp in grp_subj.groupby('block'):
                y    = grp[behav].values
                x    = grp[pupil].values
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() < 4:
                    continue
                x_ = x[mask]; y_ = y[mask]
                X  = np.column_stack([np.ones(mask.sum()), x_, x_**2])
                coefs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
                block_coefs.append(coefs)  # intercept, linear, quad
            if block_coefs:
                mean_coefs = np.mean(block_coefs, axis=0)
                all_x = grp_subj[pupil].dropna().values
                subject_fits[behav].append({'x': all_x, 'coefs': mean_coefs})
    return subject_fits


def get_quadratic_controlled_fits(df, pupil):
    subject_fits = {b: [] for b in BEHAVIOR_COLS}
    for subj, grp_subj in df.groupby('participant'):
        for behav in BEHAVIOR_COLS:
            block_coefs = []
            for block, grp in grp_subj.groupby('block'):
                y    = grp[behav].values
                x    = grp[pupil].values
                win  = grp['window'].values
                mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(win)
                if mask.sum() < 5:
                    continue
                x_ = x[mask]; y_ = y[mask]; w_ = win[mask]
                X  = np.column_stack([np.ones(mask.sum()), w_, x_, x_**2])
                coefs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
                block_coefs.append([coefs[0], coefs[2], coefs[3]])  # intercept, linear, quad
            if block_coefs:
                mean_coefs = np.mean(block_coefs, axis=0)
                all_x = grp_subj[pupil].dropna().values
                subject_fits[behav].append({'x': all_x, 'coefs': mean_coefs})
    return subject_fits


def plot_panel(ax, subject_fits, title, hv, hv_large):
    for behav in BEHAVIOR_COLS:
        fits  = subject_fits[behav]
        color = COLORS[behav]

        all_x   = np.concatenate([f['x'] for f in fits])
        x_range = np.linspace(all_x.min(), all_x.max(), 300)

        mean_c = np.mean([f['coefs'] for f in fits], axis=0)
        y_line = mean_c[0] + mean_c[1]*x_range + mean_c[2]*x_range**2

        ax.plot(x_range, y_line, color=color, linewidth=2.5, zorder=3)

    ax.set_ylim(-0.5, 2)
    ax.set_xlim(-3, 3)
    ax.set_title(title, fontproperties=hv_large, pad=10)
    ax.set_xlabel('Pupil diameter (z)', color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.set_ylabel('Behavior (z)', color='black', labelpad=14,
                  fontproperties=hv_large)
    ax.tick_params(axis='both', colors='black', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(hv)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')


def main():
    df = pd.read_csv(DATA_PATH)

    pupil = 'baseline_z'

    uncontrolled = get_quadratic_uncontrolled_fits(df, pupil)
    controlled   = get_quadratic_controlled_fits(df, pupil)

    sns.set_style("ticks")
    mpl.rcParams['font.family']     = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    hv       = fm.FontProperties(family='Helvetica', size=14)
    hv_large = fm.FontProperties(family='Helvetica', size=16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_panel(ax1, uncontrolled, '',          hv, hv_large)
    plot_panel(ax2, controlled, '',    hv, hv_large)

    ax2.set_ylabel('')

    sns.despine(trim=True, offset=10)
    plt.tight_layout(w_pad=3)
    plt.savefig('vdb_quadratic_uncontrolled_vs_controlled.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()