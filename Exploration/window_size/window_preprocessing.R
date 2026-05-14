library(tidyverse)

data <- read.csv("C:/Users/lucij/Desktop/Leiden/Year 2/Thesis Project/2024_data/combined_dataset_elapsed.csv")

n_trials <- 600

window_widths <- c(12, 20, 28, 36, 44, 52, 60, 68, 76, 84,
                   92, 100, 108, 116, 124, 132, 140, 148)

step_sizes <- round(0.333 * window_widths)

out_dir <- "C:/Users/lucij/Desktop/Leiden/Year 2/Thesis Project/2024_data/window_final"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

compute_fa_windows <- function(subject_data, window_width, step_size) {

  subject_data <- subject_data[order(subject_data$trial_number), ]

  trials   <- subject_data$trial_number
  feedback <- subject_data$feedbackType
  rt       <- subject_data$response_time
  pupil    <- subject_data$baseline_pupil
  timeout  <- subject_data$timeout
  t_ended  <- subject_data$trial_ended

  quintile_threshold <- quantile(rt, 0.80, na.rm = TRUE)
  is_slowest         <- rt > quintile_threshold
  subject_rt_mean    <- mean(rt, na.rm = TRUE)

  window_starts <- seq(1, n_trials - window_width + 1, by = step_size)

  results <- lapply(window_starts, function(s) {

    idx <- which(trials >= s & trials < s + window_width)

    if (length(idx) != window_width) return(NULL)

    wl <- t_ended[idx[which.max(trials[idx])]]

    data.frame(
      window_start     = s,
      window_time      = wl,
      fa_rate          = mean(feedback[idx] == -1, na.rm = TRUE),
      timeout_rate     = mean(timeout[idx] == 1, na.rm = TRUE),
      slowest_quintile = mean(is_slowest[idx], na.rm = TRUE),
      RT_avg           = mean(rt[idx], na.rm = TRUE),
      rtcv             = if (length(idx) < 2) NA
                         else sd(rt[idx], na.rm = TRUE) / subject_rt_mean,
      baseline         = mean(pupil[idx], na.rm = TRUE),
      derivative       = if (length(idx) < 2) NA
                         else mean(diff(pupil[idx]), na.rm = TRUE)
    )
  })

  results <- bind_rows(results)
  results$window       <- seq_len(nrow(results))
  results$instructions <- unique(subject_data$instructions)

  results %>% select(window, instructions, window_start, window_time,
                     fa_rate, timeout_rate, slowest_quintile,
                     RT_avg, rtcv, baseline, derivative)
}

cols_to_zscore <- c("fa_rate", "timeout_rate", "slowest_quintile",
                    "RT_avg", "rtcv", "baseline", "derivative")

for (i in seq_along(window_widths)) {

  ww <- window_widths[i]
  ss <- step_sizes[i]

  cat(sprintf("Running window_width = %d, step_size = %d ...\n", ww, ss))

  fa_windows_df <- data %>%
    group_by(subject) %>%
    group_modify(~ compute_fa_windows(.x, window_width = ww, step_size = ss)) %>%
    ungroup()

  cat(sprintf("  Windows per subject: %s\n",
              paste(unique(table(fa_windows_df$subject)), collapse = ", ")))

  fa_windows_df <- fa_windows_df %>%
    group_by(subject) %>%
    mutate(across(all_of(cols_to_zscore),
                  ~ scale(.x)[, 1],
                  .names = "{.col}_z")) %>%
    ungroup()

  out_file <- file.path(out_dir, sprintf("window%d.csv", ww))
  write.csv(fa_windows_df, out_file, row.names = FALSE)
  cat(sprintf("  Saved: %s\n", out_file))
}

cat("\nAll done. Files saved to:\n", out_dir, "\n")