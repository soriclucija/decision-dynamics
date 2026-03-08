library(tidyverse)

# Load data
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
  
  quintile_threshold <- quantile(rt, 0.80, na.rm = TRUE)
  is_slowest         <- rt > quintile_threshold
  
  subject_rt_mean       <- mean(rt, na.rm = TRUE)
  subject_rt_mean_05_02 <- mean(rt[contrast %in% c(0.02, 0.05)], na.rm = TRUE)
  
  window_starts <- seq(1, n_trials - window_width + 1, by = step_size)
  
  data.frame(
    window = seq_along(window_starts),
    instructions = unique(subject_data$instructions),
    
    fa_rate = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      mean(feedback[idx] == -1, na.rm = TRUE)
    }),
    
    fa_rate_05_02 = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width & contrast %in% c(0.02, 0.05))
      if (length(idx) == 0) return(NA)
      mean(feedback[idx] == -1, na.rm = TRUE)
    }),
    
    slowest_quintile = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      if (length(idx) == 0) return(NA)
      mean(is_slowest[idx], na.rm = TRUE)
    }),
    
    slowest_quintile_05_02 = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width & contrast %in% c(0.02, 0.05))
      if (length(idx) == 0) return(NA)
      mean(is_slowest[idx], na.rm = TRUE)
    }),
    
    RT_avg = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      if (length(idx) == 0) return(NA)
      mean(rt[idx], na.rm = TRUE)
    }),
    
    RT_avg_05_02 = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width & contrast %in% c(0.02, 0.05))
      if (length(idx) == 0) return(NA)
      mean(rt[idx], na.rm = TRUE)
    }),
    
    rtcv = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      if (length(idx) == 0) return(NA)
      sd(rt[idx], na.rm = TRUE) / subject_rt_mean
    }),
    
    rtcv_05_02 = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width & contrast %in% c(0.02, 0.05))
      if (length(idx) == 0) return(NA)
      sd(rt[idx], na.rm = TRUE) / subject_rt_mean_05_02
    }),
    
    baseline = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      if (length(idx) == 0) return(NA)
      mean(pupil[idx], na.rm = TRUE)
    }),
    
    baseline_05_02 = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width & contrast %in% c(0.02, 0.05))
      if (length(idx) == 0) return(NA)
      mean(pupil[idx], na.rm = TRUE)
    }),
    
    derivative = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width)
      if (length(idx) < 2) return(NA)
      mean(diff(pupil[idx]), na.rm = TRUE)
    }),
    
    derivative_05_02 = sapply(window_starts, function(s) {
      idx <- which(trials >= s & trials < s + window_width & contrast %in% c(0.02, 0.05))
      if (length(idx) < 2) return(NA)
      mean(diff(pupil[idx]), na.rm = TRUE)
    })
  )
}

# all subject
fa_windows_df <- data %>%
  group_by(subject) %>%
  group_modify(~ compute_fa_windows(.x)) %>%
  ungroup()


# Z-score all measures per subject
cols_to_zscore <- c("fa_rate", "fa_rate_05_02", "slowest_quintile", "slowest_quintile_05_02",
                    "RT_avg", "RT_avg_05_02", "rtcv", "rtcv_05_02",
                    "baseline", "baseline_05_02", "derivative", "derivative_05_02")

fa_windows_df <- fa_windows_df %>%
  group_by(subject) %>%
  mutate(across(all_of(cols_to_zscore), ~ scale(.x)[,1], .names = "{.col}_z")) %>%
  ungroup()

write.csv(fa_windows_df, "C:/Users/lucij/Desktop/Leiden/Year 2/Thesis Project/2024_data/replication_processing.csv", row.names = FALSE)

cat("Saved to replication_processing.csv\n")