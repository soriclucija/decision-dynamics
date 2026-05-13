import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\timed_replication_processed.csv")
df = df[df['instructions'] == 0]

pupil_cols    = ['baseline_z', 'derivative_z']
behavior_cols = ['fa_rate_z', 'slowest_quintile_z', 'RT_avg_z', 'rtcv_z']

# uncontrolled model 
regression_results = []

for pupil in pupil_cols:
    for behav in behavior_cols:
        linear_coefs = []
        quad_coefs   = []
        for subj, grp in df.groupby('subject'):
            y = grp[behav].values
            x = grp[pupil].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 4:
                continue
            x_lin  = x[mask]
            x_quad = x[mask] ** 2
            y_     = y[mask]
            X = np.column_stack([np.ones(mask.sum()), x_lin, x_quad])
            coefs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
            linear_coefs.append(coefs[1])
            if pupil == 'baseline_z':
                quad_coefs.append(coefs[2])

        linear_coefs = np.array(linear_coefs)
        n = len(linear_coefs)

        t_lin, p_lin_two = stats.ttest_1samp(linear_coefs, popmean=0)
        if pupil == 'derivative_z':
            p_lin    = p_lin_two
            tail_lin = 'two-tailed'
        else:
            p_lin    = p_lin_two / 2 if t_lin < 0 else 1 - p_lin_two / 2
            tail_lin = 'one-tailed'

        row = {
            'model':       'uncontrolled',
            'pupil':       pupil,
            'behavior':    behav,
            'n_subjects':  n,
            'mean_linear': round(linear_coefs.mean(), 4),
            't_linear':    round(t_lin, 4),
            'p_linear':    round(p_lin, 4),
            'tail_linear': tail_lin,
            'mean_quad':   np.nan,
            't_quad':      np.nan,
            'p_quad':      np.nan,
            'tail_quad':   np.nan,
        }

        if pupil == 'baseline_z':
            quad_coefs = np.array(quad_coefs)
            t_quad, p_quad_two = stats.ttest_1samp(quad_coefs, popmean=0)
            p_quad    = p_quad_two / 2 if t_quad > 0 else 1 - p_quad_two / 2
            row.update({
                'mean_quad': round(quad_coefs.mean(), 4),
                't_quad':    round(t_quad, 4),
                'p_quad':    round(p_quad, 4),
                'tail_quad': 'one-tailed (positive, Yerkes-Dodson)',
            })

        regression_results.append(row)

# controlled model (window_time as covariate, this is the time participants finished the last trial in the window, so it is participant and window specific)
for pupil in pupil_cols:
    for behav in behavior_cols:
        linear_coefs = []
        quad_coefs   = []
        for subj, grp in df.groupby('subject'):
            y   = grp[behav].values
            x   = grp[pupil].values
            win = grp['window_time'].values
            mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(win)
            if mask.sum() < 5:
                continue
            x_lin  = x[mask]
            x_quad = x[mask] ** 2
            w      = win[mask]
            y_     = y[mask]
            X = np.column_stack([np.ones(mask.sum()), w, x_lin, x_quad])
            coefs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
            linear_coefs.append(coefs[2])
            if pupil == 'baseline_z':
                quad_coefs.append(coefs[3])

        linear_coefs = np.array(linear_coefs)
        n = len(linear_coefs)

        t_lin, p_lin_two = stats.ttest_1samp(linear_coefs, popmean=0)
        if pupil == 'derivative_z':
            p_lin    = p_lin_two
            tail_lin = 'two-tailed'
        else:
            p_lin    = p_lin_two / 2 if t_lin > 0 else 1 - p_lin_two / 2
            tail_lin = 'one-tailed (positive, Yerkes-Dodson)'

        row = {
            'model':       'controlled',
            'pupil':       pupil,
            'behavior':    behav,
            'n_subjects':  n,
            'mean_linear': round(linear_coefs.mean(), 4),
            't_linear':    round(t_lin, 4),
            'p_linear':    round(p_lin, 4),
            'tail_linear': tail_lin,
            'mean_quad':   np.nan,
            't_quad':      np.nan,
            'p_quad':      np.nan,
            'tail_quad':   np.nan,
        }

        if pupil == 'baseline_z':
            quad_coefs = np.array(quad_coefs)
            t_quad, p_quad_two = stats.ttest_1samp(quad_coefs, popmean=0)
            p_quad    = p_quad_two / 2 if t_quad > 0 else 1 - p_quad_two / 2
            row.update({
                'mean_quad': round(quad_coefs.mean(), 4),
                't_quad':    round(t_quad, 4),
                'p_quad':    round(p_quad, 4),
                'tail_quad': 'one-tailed (positive, Yerkes-Dodson)',
            })

        regression_results.append(row)

reg_df = pd.DataFrame(regression_results)

# BH correction separately within H3a (uncontrolled) and H3b (controlled) ─
# baseline: linear + quad p-values; derivative: linear only (as in van den Brink et al., 2016)
for model_name in ['uncontrolled', 'controlled']:
    idx = reg_df['model'] == model_name
    subset = reg_df[idx]

    p_linear = subset['p_linear'].values
    p_quad   = subset.loc[subset['pupil'] == 'baseline_z', 'p_quad'].values
    all_p    = np.concatenate([p_linear, p_quad])

    reject, p_corrected, _, _ = multipletests(all_p, method='fdr_bh')

    n_lin = len(p_linear)
    reg_df.loc[idx, 'p_linear_corrected'] = p_corrected[:n_lin].round(4)
    reg_df.loc[idx, 'sig_linear']         = reject[:n_lin]

    baseline_idx = idx & (reg_df['pupil'] == 'baseline_z')
    reg_df.loc[baseline_idx, 'p_quad_corrected'] = p_corrected[n_lin:].round(4)
    reg_df.loc[baseline_idx, 'sig_quad']         = reject[n_lin:]

# results 
for model_name in ['uncontrolled', 'controlled']:
    subset = reg_df[reg_df['model'] == model_name]
    label  = "" if model_name == 'uncontrolled' else " (controlling for time-on-task)"

    print(f"\nLinear coefficients{label}:")
    print(subset[['pupil', 'behavior', 'n_subjects', 'mean_linear',
                  't_linear', 'p_linear', 'p_linear_corrected',
                  'tail_linear', 'sig_linear']].to_string(index=False))

    print(f"\nQuadratic coefficients (baseline only){label}:")
    print(subset[subset['pupil'] == 'baseline_z'][
        ['pupil', 'behavior', 'n_subjects', 'mean_quad',
         't_quad', 'p_quad', 'p_quad_corrected',
         'tail_quad', 'sig_quad']
    ].to_string(index=False))