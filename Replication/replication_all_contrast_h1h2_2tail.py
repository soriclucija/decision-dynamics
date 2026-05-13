import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

df = pd.read_csv(r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\timed_replication_processed.csv")

df = df[df['instructions'] == 1]

z_cols = [col for col in df.columns if col.endswith('_z')]

# time-on-task slopes for behavior + pupil
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
    t_stat, p_val = stats.ttest_1samp(col_slopes, popmean=0)

    results.append({
        "measure":     col,
        "n_subjects":  len(col_slopes),
        "mean_slope":  round(col_slopes.mean(), 4),
        "t_statistic": round(t_stat, 4),
        "p_value":     round(p_val, 4),
        "tail":        "two-tailed",
    })

results_df = pd.DataFrame(results)

# baseline vs derivative correlation
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
t_stat_corr, p_val_corr = stats.ttest_1samp(fisher_zs, popmean=0)
n = len(fisher_zs)
mean_r = np.tanh(np.mean(fisher_zs))

corr_row = pd.DataFrame([{
    "measure":     "baseline_x_derivative_correlation",
    "n_subjects":  n,
    "mean_slope":  round(mean_r, 4),
    "t_statistic": round(t_stat_corr, 4),
    "p_value":     round(p_val_corr, 4),
    "tail":        "two-tailed",
}])

results_df = pd.concat([results_df, corr_row], ignore_index=True)

# assign hypothesis families
def assign_family(measure):
    if 'baseline' in measure or 'derivative' in measure:
        return 'H2'
    return 'H1'

results_df['family'] = results_df['measure'].apply(assign_family)

# apply BH correction separately within each family
corrected_rows = []
for family, grp in results_df.groupby('family'):
    reject, p_corrected, _, _ = multipletests(grp['p_value'], method='fdr_bh')
    grp = grp.copy()
    grp['p_corrected'] = p_corrected.round(4)
    grp['significant'] = reject
    corrected_rows.append(grp)

results_df = pd.concat(corrected_rows).sort_index()

print(results_df.to_string(index=False))