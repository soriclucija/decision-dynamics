import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(
    r"C:\Users\lucij\Documents\combined_behavior_data_VDB.csv"
)

pupil_cols = ['baseline_z', 'derivative_z']
behavior_cols = ['fa_rate_z', 'slowest_quintile_z', 'RT_avg_z', 'rtcv_z']

def get_coefs(grp, xcol, ycol, control_col=None):

    y = grp[ycol].values
    x = grp[xcol].values

    if control_col is not None:
        w = grp[control_col].values
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(w)
        if mask.sum() < 5:
            return np.nan, np.nan

        X = np.column_stack([
            np.ones(mask.sum()),
            w[mask],
            x[mask],
            x[mask] ** 2
        ])

        coefs, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
        return coefs[2], coefs[3]  # linear, quad (pupil effects)

    else:
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 4:
            return np.nan, np.nan

        X = np.column_stack([
            np.ones(mask.sum()),
            x[mask],
            x[mask] ** 2
        ])

        coefs, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
        return coefs[1], coefs[2]

def compute_all(control_col=None):

    results = []

    for pupil in pupil_cols:
        for behav in behavior_cols:

            rows = []

            for (subj, block), grp in df.groupby(['participant', 'block']):
                lin, quad = get_coefs(grp, pupil, behav, control_col)
                rows.append([subj, block, lin, quad])

            tmp = pd.DataFrame(rows, columns=[
                'participant', 'block', 'linear', 'quad'
            ])

            subj_avg = tmp.groupby('participant')[['linear', 'quad']].mean()

            linear = subj_avg['linear'].dropna().values
            quad   = subj_avg['quad'].dropna().values

            n = len(linear)

            t_lin, p_lin_two   = stats.ttest_1samp(linear, 0)
            t_quad, p_quad_two = stats.ttest_1samp(quad, 0)

            if pupil == 'derivative_z':
                p_lin_final  = p_lin_two
                p_quad_final = p_quad_two
                tail_lin     = 'two-tailed'
                tail_quad    = 'two-tailed'
            else:
                p_lin_final  = p_lin_two  / 2 if t_lin  < 0 else 1 - p_lin_two  / 2
                p_quad_final = p_quad_two / 2 if t_quad > 0 else 1 - p_quad_two / 2
                tail_lin     = 'one-tailed'
                tail_quad    = 'one-tailed (Yerkes-Dodson)'

            results.append({
                'pupil':        pupil,
                'behavior':     behav,
                'n_subjects':   n,

                'mean_linear':  round(np.nanmean(linear), 4),
                't_linear':     round(t_lin, 4),
                'p_linear':     round(p_lin_final, 4),
                'tail_linear':  tail_lin,
                'sig_linear':   p_lin_final < 0.05,

                'mean_quad':    round(np.nanmean(quad), 4),
                't_quad':       round(t_quad, 4),
                'p_quad':       round(p_quad_final, 4),
                'tail_quad':    tail_quad,
                'sig_quad':     p_quad_final < 0.05,
            })

    return pd.DataFrame(results)

# three models - one without regressing out time-on-task, one controlling for window (proxy for time-on-task), and one controlling for model-predicted time-on-task (as they did)
reg_df   = compute_all(control_col=None)
ctrl_df  = compute_all(control_col='window')
model_df = compute_all(control_col='model_time_on_task')

print("\n=== SIMPLE MODEL ===")
print(reg_df.to_string(index=False))

print("\n=== CONTROLLED FOR WINDOW (time-on-task proxy) ===")
print(ctrl_df.to_string(index=False))

print("\n=== CONTROLLED FOR MODEL_TIME_ON_TASK ===")
print(model_df.to_string(index=False))