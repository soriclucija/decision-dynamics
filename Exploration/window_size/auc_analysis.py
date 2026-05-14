import pandas as pd
from pathlib import Path
from openpyxl import Workbook
 
from auc_utils import (
    run_regressions,
    make_group_summary,
    compute_auc_vectorized,
    plot_individual_quad_auc,
)
 
CSV_FOLDER  = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\window_final"
OUTPUT_FILE = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\window_final\regression_results.xlsx"
 
WINDOW_SIZE_MAP = {
    "window12.csv":  12,
    "window20.csv":  20,
    "window28.csv":  28,
    "window36.csv":  36,
    "window44.csv":  44,
    "window50.csv":  50,
    "window52.csv":  52,
    "window60.csv":  60,
    "window68.csv":  68,
    "window76.csv":  76,
    "window84.csv":  84,
    "window92.csv":  92,
    "window100.csv": 100,
    "window108.csv": 108,
    "window116.csv": 116,
    "window124.csv": 124,
    "window132.csv": 132,
    "window140.csv": 140,
    "window148.csv": 148,
}
 
csv_files = sorted(Path(CSV_FOLDER).glob('*.csv'))
 
all_subj_plain = []
 
for csv_path in csv_files:
    filename = csv_path.name
 
    if filename not in WINDOW_SIZE_MAP:
        continue
 
    window_size = WINDOW_SIZE_MAP[filename]
    df = pd.read_csv(csv_path)
 
    print(f"Processing {filename} (window_size={window_size}) ...")
 
    all_subj_plain.append(
        run_regressions(df, filename, window_size, controlled=True) #change for uncontrolled
    )
 
subj_plain = pd.concat(all_subj_plain, ignore_index=True)
 
auc_subj_df, auc_summary_df = compute_auc_vectorized(subj_plain)
 
print("\n=== AUC SUMMARY ===")
print(auc_summary_df.round(4).to_string(index=False))
 
print("\n=== AUC PER SUBJECT ===")
print(auc_subj_df.round(4).to_string(index=False))
 
reg_df = make_group_summary(subj_plain)
 
wb = Workbook()
ws = wb.active
ws.title = "Regressions"
 
for r in reg_df.round(4).itertuples(index=False):
    ws.append(r)
 
wb.save(OUTPUT_FILE)
print(f"\nSaved → {OUTPUT_FILE}")
 
plot_individual_quad_auc(subj_plain, out_dir=CSV_FOLDER)