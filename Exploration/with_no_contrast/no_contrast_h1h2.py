import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\no_contrast_processing.csv")

df = df[df['instructions'] == 1]

z_cols = [col for col in df.columns if col.endswith('_z')]

#time-on-task slopes for behavior + pupil
def fit_slope(series):
    x = np.arange(len(series))
    mask = ~np.isnan(series)
    if mask.sum() < 2:
        return np.nan
    slope, _, _, _, _ = stats.linregress(x[mask], series[mask])
    return slope

slopes_df = df.groupby('subject')[z_cols].apply(
    lambda grp: grp.apply(fit_slope)
).reset_index()

results = []

for col in z_cols:
    col_slopes = slopes_df[col].dropna()
    t_stat, p_two_tailed = stats.ttest_1samp(col_slopes, popmean=0)
    # two-tailed for baseline and derivative measures, one-tailed for the rest (as in van den Brink et al., 2016)
    if any(x in col for x in ['baseline', 'derivative']):
        p_val = p_two_tailed
        tail = 'two-tailed'
    else:
        p_val = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
        tail = 'one-tailed'
        
    results.append({
        "measure":      col,
        "n_subjects":   len(col_slopes),
        "mean_slope":   round(col_slopes.mean(), 4),
        "t_statistic":  round(t_stat, 4),
        "p_value":      round(p_val, 4),
        "tail":         tail,
        "significant":  p_val < 0.05
    })
    results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# baseline and derivative pupil correlation
fisher_zs = []

for subj, grp in df.groupby('subject'):
    b = grp['baseline_z'].values
    d = grp['derivative_z'].values
    mask = ~np.isnan(b) & ~np.isnan(d)
    if mask.sum() < 2:
        continue
    r, _ = stats.pearsonr(b[mask], d[mask])
    fisher_zs.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))

fisher_zs = np.array(fisher_zs)
t_stat, p_val = stats.ttest_1samp(fisher_zs, popmean=0)
n      = len(fisher_zs)
mean_r = np.tanh(np.mean(fisher_zs))
r2     = mean_r ** 2

print(f"\n{'─'*55}")
print(f"Baseline vs Derivative correlation (across windows)")
print(f"N subjects:{n}")
print(f"Mean r:{mean_r:.4f}")
print(f"R²:{r2:.4f} ({r2*100:.2f}%)")
print(f"t({n-1}) = {t_stat:.2f}, p = {p_val:.4f} (two-tailed)")
