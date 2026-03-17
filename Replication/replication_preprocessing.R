library(tidyverse)

data <- read.csv("C:/Users/lucij/Desktop/Leiden/Year 2/Thesis Project/2024_data/combined_dataset.csv")

window_width <- 50
step_size    <- 15
n_trials     <- 600

compute_fa_windows <- function(subject_data) {
  
  subject_data <- subject_data[order(subject_data$trial_number), ]

  trials   <- subject_data$trial_number
  feedback <- subject_data$feedbackType
  contrast <- subject_data$stimContrast
  rt       <- subject_data$response_time
  pupil    <- subject_data$baseline_pupil
  timeout  <- subject_data$timeout
  
  quintile_threshold <- quantile(rt, 0.80, na.rm = TRUE)
  is_slowest         <- rt > quintile_threshold
  
  subject_rt_mean       <- mean(rt, na.rm = TRUE)
  
  window_starts <- seq(1, n_trials - window_width + 1, by = step_size)
  
  data.frame(
    window = seq_along(window_starts),
    instructions = unique(subject_data$instructions),
    
    fa_rate = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      mean(feedback[valid_idx] == -1, na.rm = TRUE)
    }),
    
    timeout_rate = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      mean(timeout[valid_idx] == 1, na.rm = TRUE)
    }),
    
    slowest_quintile = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      if (length(valid_idx) == 0) return(NA)
      mean(is_slowest[valid_idx], na.rm = TRUE)
    }),
    
    RT_avg = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      if (length(valid_idx) == 0) return(NA)
      mean(rt[valid_idx], na.rm = TRUE)
    }),
    
    rtcv = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      if (length(valid_idx) == 0) return(NA)
      sd(rt[valid_idx], na.rm = TRUE) / subject_rt_mean
    }),
    
    baseline = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      if (length(valid_idx) == 0) return(NA)
      mean(pupil[valid_idx], na.rm = TRUE)
    }),
    
    derivative = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      valid_idx <- idx[contrast[idx] != 0]
      if (length(valid_idx) < 2) return(NA)
      mean(diff(pupil[valid_idx]), na.rm = TRUE)
    })
  )
}

# all subject
fa_windows_df <- data %>%
  group_by(subject) %>%
  group_modify(~ compute_fa_windows(.x)) %>%
  ungroup()


# Z-score all measures per subject
cols_to_zscore <- c("fa_rate", "timeout_rate", "slowest_quintile",
                    "RT_avg", "rtcv", 
                    "baseline", "derivative")

fa_windows_df <- fa_windows_df %>%
  group_by(subject) %>%
  mutate(across(all_of(cols_to_zscore), ~ scale(.x)[,1], .names = "{.col}_z")) %>%
  ungroup()

write.csv(fa_windows_df, "C:/Users/lucij/Desktop/Leiden/Year 2/Thesis Project/2024_data/replication_processing.csv", row.names = FALSE)

cat("Saved to replication_processing.csv\n")