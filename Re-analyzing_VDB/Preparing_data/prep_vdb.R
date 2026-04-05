library(dplyr)
library(tidyr)
library(readr)

process_file <- function(file_path, measure_name) {
  
  df <- read.table(file_path, header = FALSE)
  colnames(df)[1:2] <- c("participant", "block")
  colnames(df)[-(1:2)] <- paste0("window_", seq_len(ncol(df) - 2))
  
  df_long <- df %>%
    pivot_longer(
      cols = starts_with("window_"),
      names_to = "window",
      values_to = measure_name
    ) %>%
    mutate(window = as.numeric(gsub("window_", "", window)))
  
  return(df_long)
}

fa_rate <- process_file(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/behavior/False_alarms.txt",
  "fa_rate_z"
)
rtcv <- process_file(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/behavior/RTCV.txt",
  "rtcv_z"
)
RT_avg <- process_file(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/behavior/RTs.txt",
  "RT_avg_z"
)
slowest_quintile <- process_file(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/behavior/Slow_quintile.txt",
  "slowest_quintile_z"
)
derivative <- process_file(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/pupil/Derivative.txt",
  "derivative_z"
)
diameter <- process_file(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/pupil/Diameter.txt",
  "baseline_z"
)

final_df <- fa_rate %>%
  full_join(rtcv,             by = c("participant", "block", "window")) %>%
  full_join(RT_avg,           by = c("participant", "block", "window")) %>%
  full_join(slowest_quintile, by = c("participant", "block", "window")) %>%
  full_join(derivative,       by = c("participant", "block", "window")) %>%
  full_join(diameter,         by = c("participant", "block", "window"))

final_df <- final_df %>%
  select(participant, block, window,
         fa_rate_z, rtcv_z, RT_avg_z, slowest_quintile_z,
         derivative_z, baseline_z)

write_csv(final_df, "combined_behavior_data_VDB.csv")
