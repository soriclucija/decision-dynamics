import pandas as pd
from pathlib import Path
from openpyxl import Workbook

from auc_utils import (
    run_regressions,
    make_group_summary,
    compute_auc_vectorized, 
    plot_individual_quad_auc,
)

CSV_FOLDER  = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\window_sizes"
OUTPUT_FILE = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\regression_results.xlsx"

WINDOW_SIZE_MAP = { #add if needed
    "w5_s1_preprocessing.csv":   5,
    "w13_s5_preprocessing.csv":  13,
    "w15_s5_preprocessing.csv":  15,
    "w20_s1_preprocessing.csv":  20,
    "w25_s7_preprocessing.csv":  25,
    "w30_s5_preprocessing.csv":  30,
    "w30_9_preprocessing.csv":   30,
    "w40_s10_preprocessing.csv": 40,
    "replication_processing.csv": 50,
    "w60_s18_preprocessing.csv": 60,
    "w75_s25_preprocessing.csv": 75,
}

csv_files = sorted(Path(CSV_FOLDER).glob('*.csv'))

all_subj_plain = []

for csv_path in csv_files:
    filename = csv_path.name

    if filename not in WINDOW_SIZE_MAP:
        continue

    window_size = WINDOW_SIZE_MAP[filename]
    df = pd.read_csv(csv_path)

    all_subj_plain.append(
        run_regressions(df, filename, window_size, controlled=False) #change to False for uncontrolled
    )

subj_plain = pd.concat(all_subj_plain, ignore_index=True)

auc_subj_df, auc_summary_df = compute_auc_vectorized(subj_plain)

print("\n=== AUC SUMMARY ===")
print(auc_summary_df.round(4).to_string(index=False))

print("\n=== AUC PER SUBJECT ===")
print(auc_subj_df.round(4).to_string(index=False))

# also saved in excel
reg_df = make_group_summary(subj_plain)

wb = Workbook()
ws = wb.active
ws.title = "Regressions"

for r in reg_df.round(4).itertuples(index=False):
    ws.append(r)

wb.save(OUTPUT_FILE)

print(f"\nSaved → {OUTPUT_FILE}")

plot_individual_quad_auc(subj_plain)