import os
import pandas as pd
import numpy as np


beh_folder = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\cleaned_behavioral_data"
raw_pupil_folder = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\raw_eyelink"
processed_pupil_folder = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\processed_eyelink"
output_folder = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\combined_behavioral_eyetracking"

os.makedirs(output_folder, exist_ok=True)

# asc events
def read_asc_events(asc_path):
    events = []
    with open(asc_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0] == 'MSG':
                try:
                    time_ms = float(parts[1])
                    event_name = " ".join(parts[2:])
                    events.append((time_ms, event_name))
                except ValueError:
                    continue

    events_df = pd.DataFrame(events, columns=['time', 'event'])
    events_df['time_s'] = events_df['time'] / 1000
    return events_df

# averaging pupil measures per trial
def compute_trial_pupil_measures(pupil_df, events_df):
    trial_measures = []

    pupil_col = 'pupil_int_lp_clean' #from preprocessed files

    trial_starts = events_df[events_df['event'].str.contains('fix_cross_ONSET')].reset_index(drop=True)
    n_trials = len(trial_starts)

    for i in range(n_trials):
        start_time = trial_starts.loc[i, 'time_s']
        end_time = (
            trial_starts.loc[i + 1, 'time_s']
            if i < n_trials - 1
            else pupil_df['time'].max()
        )

        trial_dict = {'trial_number': i + 1}
        trial_events = events_df[
            (events_df['time_s'] >= start_time) &
            (events_df['time_s'] < end_time)
        ]

        def pupil_stats_between(event_start_name, event_end_name):
            try:
                start_ev = trial_events[
                    trial_events['event'].str.contains(event_start_name)
                ].iloc[0]['time_s']

                end_ev = trial_events[
                    trial_events['event'].str.contains(event_end_name)
                ].iloc[0]['time_s']

                mask = (pupil_df['time'] >= start_ev) & (pupil_df['time'] <= end_ev)
                values = pupil_df.loc[mask, pupil_col]

                return values.mean(), values.std()

            except IndexError:
                return np.nan, np.nan

        # baseline (q)
        m, sd = pupil_stats_between('fix_cross_ONSET', 'stimOn')
        trial_dict['baseline_pupil'] = m
        trial_dict['baseline_pupil_sd'] = sd

        # onsent until movement
        m, sd = pupil_stats_between('sound_trial_start_ONSET', 'moveInit')
        trial_dict['stimulus_pupil'] = m
        trial_dict['stimulus_pupil_sd'] = sd

        # response period (movement until response)
        m, sd = pupil_stats_between('moveInit', 'response')
        trial_dict['response_pupil'] = m
        trial_dict['response_pupil_sd'] = sd

        # feedback time
        m, sd = pupil_stats_between('feedback_sound_ONSET', 'feedback_sound_OFFSET')
        trial_dict['feedback_pupil'] = m
        trial_dict['feedback_pupil_sd'] = sd

        # post-baseline
        m, sd = pupil_stats_between('feedback_sound_OFFSET', 'fix_cross_ONSET')
        trial_dict['post_baseline_pupil'] = m
        trial_dict['post_baseline_pupil_sd'] = sd

        trial_measures.append(trial_dict)

    return pd.DataFrame(trial_measures)

# loop through subjects
for beh_file in os.listdir(beh_folder):
    if not beh_file.endswith('.csv'):
        continue

    subj_id = beh_file[:3]
    print(f"Processing subject {subj_id}...")

    beh_df = pd.read_csv(os.path.join(beh_folder, beh_file))

    proc_file = [f for f in os.listdir(processed_pupil_folder) if f.startswith(subj_id)]
    if not proc_file:
        print(f"No processed pupil file for {subj_id}, skipping...")
        continue
    pupil_df = pd.read_hdf(os.path.join(processed_pupil_folder, proc_file[0]))

    asc_file = [f for f in os.listdir(raw_pupil_folder) if f.startswith(subj_id)]
    if not asc_file:
        print(f"No raw ASC file for {subj_id}, skipping...")
        continue
    events_df = read_asc_events(os.path.join(raw_pupil_folder, asc_file[0]))

    # align times
    pupil_start = pupil_df['time'].iloc[0]
    first_fix_event = events_df[
        events_df['event'].str.contains('fix_cross_ONSET')
    ]['time_s'].iloc[0]

    time_offset = first_fix_event - pupil_start
    events_df['time_s'] -= time_offset

    # compute pupil measures
    trial_pupil_df = compute_trial_pupil_measures(pupil_df, events_df)
    print(f"Saved {trial_pupil_df.shape[0]} trials for subject {subj_id}.")

    # merge
    combined_df = pd.merge(
        beh_df,
        trial_pupil_df,
        on='trial_number',
        how='left'
    )

    # save
    out_path = os.path.join(output_folder, f"{subj_id}_combined.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")