library(tidyverse)

model_tot <- read.table(
  "C:/Users/lucij/Downloads/vandenBrinketal2016PONE/processed_data/pupil/Model_time_on_task.txt",
  header = FALSE
)

combined <- read.csv("C:/Users/lucij/Documents/combined_behavior_data_VDB.csv")

stopifnot(nrow(model_tot) == 36)

model_tot_vec <- model_tot$V1  

combined <- combined %>%
  mutate(model_time_on_task = model_tot_vec[window])

write.csv(combined, "C:/Users/lucij/Documents/combined_behavior_data_VDB.csv", row.names = FALSE)

cat("Done! Column 'model_time_on_task' added.\n")
cat("Preview:\n")
print(head(combined[, c("window", "model_time_on_task")]))