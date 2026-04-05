import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\lucij\Documents\combined_behavior_data_VDB.csv"
)


# ------------------------------------------------------------
# DEFINE Z-COLUMNS
# ------------------------------------------------------------
z_cols = [col for col in df.columns if col.endswith('_z')]

# ------------------------------------------------------------
# SLOPE FUNCTION (time-on-task within block)
# ------------------------------------------------------------
def fit_slope(series):
    x = np.arange(len(series))
    mask = ~np.isnan(series)

    if mask.sum() < 2:
        return np.nan

    slope, _, _, _, _ = stats.linregress(x[mask], series[mask])
    return slope

# ------------------------------------------------------------
# COMPUTE SLOPES (participant × block)
# ------------------------------------------------------------
slopes_df = df.groupby(['participant', 'block'])[z_cols].apply(
    lambda grp: grp.apply(fit_slope)
).reset_index()

participant_slopes = slopes_df.groupby('participant')[z_cols].mean().reset_index()

# ------------------------------------------------------------
# STATS ON SLOPES
# ------------------------------------------------------------
results = []

for col in z_cols:
    col_slopes = participant_slopes[col].dropna()

    t_stat, p_two_tailed = stats.ttest_1samp(col_slopes, popmean=0)

    # one-tailed vs two-tailed (as in van den Brink et al.)
    if any(x in col for x in ['baseline', 'derivative']):
        p_val = p_two_tailed
        tail = 'two-tailed'
    else:
        p_val = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2
        tail = 'one-tailed'

    results.append({
        "measure": col,
        "n_blocks": len(col_slopes),
        "mean_slope": round(col_slopes.mean(), 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "tail": tail,
        "significant": p_val < 0.05
    })

participant_slopes = pd.DataFrame(results)

print("\nSLOPE RESULTS")
print(participant_slopes.to_string(index=False))

# ------------------------------------------------------------
# PUPIL CORRELATION (within participant)
# baseline_z vs derivative_z across windows
# ------------------------------------------------------------
fisher_zs = []

for subj, grp in df.groupby('participant'):

    # average across blocks first (important for stability)
    sub_avg = grp.groupby('window')[['baseline_z', 'derivative_z']].mean()

    b = sub_avg['baseline_z'].values
    d = sub_avg['derivative_z'].values

    mask = ~np.isnan(b) & ~np.isnan(d)

    if mask.sum() < 2:
        continue

    r, _ = stats.pearsonr(b[mask], d[mask])
    fisher_zs.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))

fisher_zs = np.array(fisher_zs)

# ------------------------------------------------------------
# GROUP STATISTICS
# ------------------------------------------------------------
t_stat, p_val = stats.ttest_1samp(fisher_zs, popmean=0)

n = len(fisher_zs)
mean_r = np.tanh(np.mean(fisher_zs))
r2 = mean_r ** 2

print("\n" + "─" * 55)
print("Baseline vs Derivative correlation (across participants)")
print(f"N subjects: {n}")
print(f"Mean r: {mean_r:.4f}")
print(f"R²: {r2:.4f} ({r2*100:.2f}%)")
print(f"t({n-1}) = {t_stat:.2f}, p = {p_val:.4f} (two-tailed)")