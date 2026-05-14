import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

COLORS = {0: "#daa800", 1: "#840000"}
LABELS = {0: "Min.",    1: "Full"}

AUC_DIRECTION = {
    ('baseline_z',   'linear'): 'negative',
    ('derivative_z', 'linear'): 'positive',
    ('baseline_z',   'quad'):   'positive',  # Yerkes-Dodson, flipped: larger values = worse performance
}

pupil_cols    = ['baseline_z', 'derivative_z']
behavior_cols = ['fa_rate_z', 'slowest_quintile_z', 'RT_avg_z', 'rtcv_z']


def get_p_linear(coefs, pupil, condition):
    t, p_two = stats.ttest_1samp(coefs, 0)
    if condition == 0 or pupil == 'derivative_z':
        return t, p_two, 'two-tailed'
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return t, p_one, 'one-tailed (positive)'

def get_p_quad(coefs, condition):
    t, p_two = stats.ttest_1samp(coefs, 0)
    if condition == 0:
        return t, p_two, 'two-tailed'
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return t, p_one, 'one-tailed (positive)'

def get_p_auc(aucs, direction, condition):
    t, p_two = stats.ttest_1samp(aucs, 0)
    if condition == 0:
        return t, p_two, 'two-tailed'
    if direction == 'negative':
        p_one = p_two / 2 if t < 0 else 1 - p_two / 2
    else:
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return t, p_one, f"one-tailed ({direction})"


def run_regressions(df, source_label, window_size, controlled=False):
    results = []

    for condition in sorted(df['instructions'].unique()):
        df_cond = df[df['instructions'] == condition]

        for pupil in pupil_cols:
            for behav in behavior_cols:
                for subj, grp in df_cond.groupby('subject'):

                    y = grp[behav].values
                    x = grp[pupil].values

                    min_obs = 5 if controlled else 4

                    if controlled:
                        win  = grp['window'].values
                        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(win)
                    else:
                        mask = ~np.isnan(x) & ~np.isnan(y)

                    if mask.sum() < min_obs:
                        continue

                    x_lin  = x[mask]
                    x_quad = x_lin ** 2
                    y_     = y[mask]

                    if controlled:
                        w     = grp['window'].values[mask]
                        X     = np.column_stack([np.ones(len(x_lin)), w, x_lin, x_quad])
                        coefs, *_ = np.linalg.lstsq(X, y_, rcond=None)
                        lin_coef, quad_coef = coefs[2], coefs[3]
                    else:
                        X     = np.column_stack([np.ones(len(x_lin)), x_lin, x_quad])
                        coefs, *_ = np.linalg.lstsq(X, y_, rcond=None)
                        lin_coef, quad_coef = coefs[1], coefs[2]

                    results.append({
                        'source':       source_label,
                        'window_size':  window_size,
                        'condition':    condition,
                        'subject':      subj,
                        'pupil':        pupil,
                        'behavior':     behav,
                        'linear_coef':  lin_coef,
                        'quad_coef':    quad_coef if pupil != 'derivative_z' else np.nan,
                    })

    return pd.DataFrame(results)


def make_group_summary(subj_df):
    rows = []

    for keys, grp in subj_df.groupby(['source', 'window_size', 'condition', 'pupil', 'behavior']):
        source, ws, condition, pupil, behav = keys

        lin  = grp['linear_coef'].dropna().values
        quad = grp['quad_coef'].dropna().values

        if len(lin) == 0:
            continue

        t_lin, p_lin, tail_lin = get_p_linear(lin, pupil, condition)

        row = {
            'source':       source,
            'window_size':  ws,
            'condition':    condition,
            'pupil':        pupil,
            'behavior':     behav,
            'n_subjects':   len(lin),
            'mean_linear':  lin.mean(),
            't_linear':     t_lin,
            'p_linear':     p_lin,
            'tail_linear':  tail_lin,
            'sig_linear':   p_lin < 0.05,
            'mean_quad':    None,
            't_quad':       None,
            'p_quad':       None,
            'tail_quad':    None,
            'sig_quad':     None,
        }

        if pupil != 'derivative_z' and len(quad) > 0:
            t_quad, p_quad, tail_quad = get_p_quad(quad, condition)
            row.update({
                'mean_quad': quad.mean(),
                't_quad':    t_quad,
                'p_quad':    p_quad,
                'tail_quad': tail_quad,
                'sig_quad':  p_quad < 0.05,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def compute_auc_vectorized(subj_df):
    subj_avg = (
        subj_df
        .groupby(['condition', 'pupil', 'subject', 'window_size', 'behavior'])[['linear_coef', 'quad_coef']]
        .mean()
        .groupby(['condition', 'pupil', 'subject', 'window_size'])
        .mean()
        .reset_index()
        .sort_values(['condition', 'pupil', 'subject', 'window_size'])
    )

    def trapz_calc(group):
        auc_lin  = np.trapezoid(group['linear_coef'], group['window_size'])
        is_valid = group['pupil'].iloc[0] != 'derivative_z'
        auc_quad = np.trapezoid(group['quad_coef'], group['window_size']) if is_valid else np.nan
        return pd.Series({'AUC_linear': auc_lin, 'AUC_quad': auc_quad, 'n_windows': len(group)})

    auc_subj_df = (
        subj_avg
        .groupby(['condition', 'pupil', 'subject'])
        .apply(trapz_calc, include_groups=True)
        .reset_index()
    )

    def summarize_group(grp):
        cond  = grp['condition'].iloc[0]
        pupil = grp['pupil'].iloc[0]

        lin_vals  = grp['AUC_linear'].dropna()
        quad_vals = grp['AUC_quad'].dropna()

        t_lin, p_lin, tail_lin = get_p_auc(
            lin_vals.values,
            AUC_DIRECTION.get((pupil, 'linear'), 'positive'),
            cond
        )

        res = {
            'n_subjects':      len(lin_vals),
            'mean_AUC_linear': lin_vals.mean(),
            't_AUC_linear':    t_lin,
            'p_AUC_linear':    p_lin,
            'sig_AUC_linear':  p_lin < 0.05,
        }

        if not quad_vals.empty:
            t_q, p_q, tail_q = get_p_auc(
                quad_vals.values,
                AUC_DIRECTION.get((pupil, 'quad'), 'negative'),
                cond
            )
            res.update({
                'mean_AUC_quad': quad_vals.mean(),
                't_AUC_quad':    t_q,
                'p_AUC_quad':    p_q,
                'sig_AUC_quad':  p_q < 0.05,
            })

        return pd.Series(res)

    summary_df = (
        auc_subj_df
        .groupby(['condition', 'pupil'])
        .apply(summarize_group, include_groups=True)
        .reset_index()
    )

    return auc_subj_df, summary_df


def _style_ax(ax, ws_ticks, ylabel, hv, hv_large, fig, hline=False):
    """Apply shared axis styling."""
    ax.set_xticks(ws_ticks)
    ax.set_xticklabels([str(x) for x in ws_ticks], ha='center', color='black')

    for i, lbl in enumerate(ax.get_xticklabels()):
        if i % 2 == 1:
            lbl.set_transform(
                lbl.get_transform() +
                mpl.transforms.ScaledTranslation(0, -0.25, fig.dpi_scale_trans)
            )

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(hv)

    ax.set_xlabel('Window size (trials)', fontproperties=hv_large, labelpad=12)
    ax.set_ylabel(ylabel, fontproperties=hv_large, labelpad=12)
    ax.tick_params(axis='both', which='both',
                   length=2, width=1, direction='out',
                   pad=4, colors='black', labelsize=14)

    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    if hline:
        ax.axhline(y=0, color='#fdeabe', linestyle='--', linewidth=2, alpha=0.8, zorder=1)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    sns.despine(ax=ax, trim=False)

    for artist in ax.lines + ax.collections:
        artist.set_clip_on(False)

    fig.canvas.draw()

    for i, lbl in enumerate(ax.get_xticklabels()):
        if i % 2 == 1:
            lbl.set_transform(
                lbl.get_transform() +
                mpl.transforms.ScaledTranslation(0, -0.2, fig.dpi_scale_trans)
            )


def plot_individual_quad_auc(subj_df, out_dir="."):
    sns.set_style("ticks")
    mpl.rcParams['font.family']     = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']

    hv       = fm.FontProperties(family='Helvetica', size=12)
    hv_large = fm.FontProperties(family='Helvetica', size=16)

    out_dir    = Path(out_dir)
    conditions = [1, 0]

    # --- baseline_z: linear + quadratic side by side ---
    fig, (ax_lin, ax_quad) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(wspace=0.35)

    all_ws = set()

    for condition in conditions:
        subset = (
            subj_df[
                (subj_df['condition'] == condition) &
                (subj_df['pupil']     == 'baseline_z')
            ]
            .groupby('window_size')[['linear_coef', 'quad_coef']]
            .mean()
            .reset_index()
        )

        if subset.empty:
            continue

        color = COLORS[condition]
        label = LABELS[condition]

        for ax, col in [(ax_lin, 'linear_coef'), (ax_quad, 'quad_coef')]:
            ax.plot(
                subset['window_size'], subset[col],
                color=color, linewidth=3, label=label, zorder=10
            )
            ax.scatter(
                subset['window_size'], subset[col],
                color=color, s=12**2, zorder=20,
                edgecolors='white', linewidths=2, alpha=1
            )

        all_ws.update(subset['window_size'].tolist())

    ws_ticks = sorted(all_ws)

    _style_ax(ax_lin,  ws_ticks, 'Linear coefficient',    hv, hv_large, fig, hline=True)
    _style_ax(ax_quad, ws_ticks, 'Quadratic coefficient', hv, hv_large, fig, hline=False)

    handles, labels = ax_lin.get_legend_handles_labels()

    filename = out_dir / "baseline_linear_quad_auc_contr.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")