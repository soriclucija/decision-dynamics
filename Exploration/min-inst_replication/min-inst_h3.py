import pandas as pd
import numpy as np
from scipy import stats

#exploratory analyses into the relationship between pupil and behavior in min. instr. condition
#adapted from replication code, changing directionality of tests (two-tailed due to exploratory nature)

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\replication_processing.csv")
df = df[df['instructions'] == 0]

pupil_cols = ['baseline_z', 'derivative_z']
behavior_cols = ['fa_rate_z', 'slowest_quintile_z', 'RT_avg_z', 'rtcv_z']

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
            x_quad = x[mask] **2
            y_     = y[mask]
            X = np.column_stack([np.ones(mask.sum()), x_lin, x_quad])
            coefs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
            linear_coefs.append(coefs[1])
            quad_coefs.append(coefs[2])
        linear_coefs = np.array(linear_coefs)
        quad_coefs   = np.array(quad_coefs)
        n = len(linear_coefs)

        t_lin, p_lin_two = stats.ttest_1samp(linear_coefs, popmean=0)
        if pupil == 'derivative_z':
            p_lin    = p_lin_two
            tail_lin = 'two-tailed'
        else:
            p_lin    = p_lin_two
            tail_lin = 'two-tailed'

        t_quad, p_quad_two = stats.ttest_1samp(quad_coefs, popmean=0)
        if pupil == 'derivative_z':
            p_quad    = p_quad_two
            tail_quad = 'two-tailed'
        else:
            p_quad    = p_quad_two
            tail_quad = 'two-tailed'

        regression_results.append({
            'pupil':       pupil,
            'behavior':    behav,
            'n_subjects':  n,
            'mean_linear': round(linear_coefs.mean(), 4),
            't_linear':    round(t_lin, 4),
            'p_linear':    round(p_lin, 4),
            'tail_linear': tail_lin,
            'sig_linear':  p_lin < 0.05,
            'mean_quad':   round(quad_coefs.mean(), 4),
            't_quad':      round(t_quad, 4),
            'p_quad':      round(p_quad, 4),
            'tail_quad':   tail_quad,
            'sig_quad':    p_quad < 0.05,
        })

reg_df = pd.DataFrame(regression_results)

print("\nLinear coefficients:")
print(reg_df[['pupil','behavior','n_subjects','mean_linear','t_linear','p_linear','tail_linear','sig_linear']].to_string(index=False))

print("\nQuadratic coefficients:")
print(reg_df[['pupil','behavior','n_subjects','mean_quad','t_quad','p_quad','tail_quad','sig_quad']].to_string(index=False))


controlled_results = []

for pupil in pupil_cols:
    for behav in behavior_cols:
        linear_coefs = []
        quad_coefs   = []
        for subj, grp in df.groupby('subject'):
            y   = grp[behav].values
            x   = grp[pupil].values
            win = grp['window'].values  # linearly increasing time-on-task predictor
            mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(win)
            if mask.sum() < 5:  
                continue
            x_lin  = x[mask]
            x_quad = x[mask] **2
            w      = win[mask]
            y_     = y[mask]

            X = np.column_stack([np.ones(mask.sum()), w, x_lin, x_quad])
            coefs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
            linear_coefs.append(coefs[2])  
            quad_coefs.append(coefs[3])    

        linear_coefs = np.array(linear_coefs)
        quad_coefs   = np.array(quad_coefs)
        n = len(linear_coefs)

        t_lin, p_lin_two = stats.ttest_1samp(linear_coefs, popmean=0)
        if pupil == 'derivative_z':
            p_lin    = p_lin_two
            tail_lin = 'two-tailed'
        else:
            p_lin    = p_lin_two
            tail_lin = 'two-tailed'

        t_quad, p_quad_two = stats.ttest_1samp(quad_coefs, popmean=0)
        if pupil == 'derivative_z':
            p_quad    = p_quad_two
            tail_quad = 'two-tailed'
        else:
            p_quad    = p_quad_two 
            tail_quad = 'two-tailed'  

        controlled_results.append({
            'pupil':       pupil,
            'behavior':    behav,
            'n_subjects':  n,
            'mean_linear': round(linear_coefs.mean(), 4),
            't_linear':    round(t_lin, 4),
            'p_linear':    round(p_lin, 4),
            'tail_linear': tail_lin,
            'sig_linear':  p_lin < 0.05,
            'mean_quad':   round(quad_coefs.mean(), 4),
            't_quad':      round(t_quad, 4),
            'p_quad':      round(p_quad, 4),
            'tail_quad':   tail_quad,
            'sig_quad':    p_quad < 0.05,
        })

ctrl_df = pd.DataFrame(controlled_results)

print("\nLinear coefficients (controlling for time-on-task):")
print(ctrl_df[['pupil','behavior','n_subjects','mean_linear','t_linear','p_linear','tail_linear','sig_linear']].to_string(index=False))

print("\nQuadratic coefficients (controlling for time-on-task):")
print(ctrl_df[['pupil','behavior','n_subjects','mean_quad','t_quad','p_quad','tail_quad','sig_quad']].to_string(index=False))
