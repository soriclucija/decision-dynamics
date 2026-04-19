# Tracking Behavioral and Physiological Dynamics in Sustained Perceptual Decisions

Master's thesis project — School Year 2025/2026

This repository contains all code developed for the master's thesis *Tracking Behavioral and Physiological Dynamics in Sustained Perceptual Decisions*. The project investigates how behavioral and physiological signals (primarily pupillometry) evolve over the course of sustained perceptual decision-making, including a confirmatory replication of prior findings and a range of exploratory analyses.

---

## Repository Structure

```
.
├── Cleaning/                  # Combines behavioral and pupil data; cleans raw data files
├── Descriptives/              # Notebook with descriptive statistics of the sample
├── EEG_Preprocessing/         # EEG preprocessing pipeline (developed for pilot; not used in final thesis)
├── Exploration/
│   ├── min-inst_replication/  # Exploratory analysis of the minimal instructions condition
│   ├── window-size/           # Sliding window preprocessing across window sizes + AUC robustness checks
│   └── with_no_contrast/      # Analysis checking the effect of excluding no-contrast trials
├── Pupil_preprocessing/       # Preprocesses raw pupillary data into trial-averaged pupil signals
├── Re-analyzing_VDB/          # Reproduces findings from van den Brink et al. (2016)
├── Replication/               # Main confirmatory analyses of the thesis
└── Visualization/             # All figures and plots generated for the thesis
```

---

## Folder Descriptions

### `Cleaning/`
Scripts for merging behavioral response data with pupillometric recordings and applying data cleaning procedures (e.g., artifact removal, trial filtering).

### `Descriptives/`
A Jupyter notebook providing an overview of the sample's descriptive statistics, including participant demographics and task performance summaries.

### `EEG_Preprocessing/`
A preprocessing pipeline developed during an EEG pilot study. Although EEG data were ultimately not included in the final thesis, this pipeline was developed as part of the original research plan and is retained here for completeness.

### `Exploration/`

- **`min-inst_replication/`** — Exploratory analyses of the minimal instructions condition, following the same analytic steps as the confirmatory replication to allow for comparison.
- **`window-size/`** — Implements sliding window preprocessing across a range of window sizes. Includes AUC (Area Under the Curve) calculations to assess whether key findings are robust to the choice of window size.
- **`with_no_contrast/`** — Investigates how excluding no-contrast trials during preprocessing affects the results, serving as a robustness check on the data cleaning decisions.

### `Pupil_preprocessing/`
Preprocesses raw pupillary time-series data into trial-averaged pupil dilation signals ready for downstream analysis.

### `Re-analyzing_VDB/`
Code developed to reproduce the key findings reported in:
> van den Brink, R. L., et al. (2016). *...*

This serves as a baseline and sanity check for the analysis approach used in the thesis.

### `Replication/`
The core of the thesis. Contains all confirmatory (pre-registered) analyses examining behavioral and physiological dynamics in sustained perceptual decision-making.

### `Visualization/`
Scripts and notebooks used to generate all figures appearing in the thesis, including pupil time-course plots, performance metrics, and summary statistics visualizations.

---

## Context

This project was conducted as part of a master's thesis in the 2025/2026 academic year. The primary data sources are behavioral responses and pupillometric recordings collected during a sustained perceptual decision-making task. A key component of the thesis is a confirmatory replication of previously published findings, complemented by a series of exploratory analyses to assess the robustness and generalizability of the results.

---

## Reference

van den Brink, R. L., Murphy, P. R., & Nieuwenhuis, S. (2016). Pupil Diameter Tracks Lapses of Attention. *PLOS ONE, 11*(10), e0165274. https://doi.org/10.1371/journal.pone.0165274

---

## License

This project is licensed under the [MIT License](LICENSE).

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
