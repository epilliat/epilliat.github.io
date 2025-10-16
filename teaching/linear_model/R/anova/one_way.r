# Load necessary libraries for analysis and plotting 📊
library(car)
library(ggplot2)
library(dplyr)

# --- 1. Data Generation ---
# Set a seed for consistent, reproducible results
set.seed(123)

# Define parameters for the dataset
n_per_group <- 18
group_means <- c(10, 15, 18, 12) # Mean loss for each exercise type
within_sd <- sqrt(11.5)

# Create the data frame
data <- data.frame(
  Exercise = factor(rep(paste0("Exercise", 1:4), each = n_per_group)),
  Loss = c(
    rnorm(n_per_group, mean = group_means[1], sd = within_sd),
    rnorm(n_per_group, mean = group_means[2], sd = within_sd),
    rnorm(n_per_group, mean = group_means[3], sd = within_sd),
    rnorm(n_per_group, mean = group_means[4], sd = within_sd)
  )
)

# --- 2. Statistical Analysis ---
# Run ANOVA to check for any overall significant difference
reg <- lm(Loss ~ Exercise, data = data)
summary(reg)
aov_model <- aov(Loss ~ Exercise, data = data)
aov_model <- anova(reg)
cat("--- ANOVA Results ---\n")
aov_model

# Since the ANOVA is significant, run a Tukey HSD test
# to find out which specific groups differ.
tukey_results <- TukeyHSD(aov_model)
cat("\n--- Tukey HSD Pairwise Comparisons ---\n")
print(tukey_results)

# --- 3. Visualization ---
# Calculate summary stats for the plot (mean and standard error)
plot_data <- data %>%
  group_by(Exercise) %>%
  summarise(
    mean_Loss = mean(Loss),
    se_Loss = sd(Loss) / sqrt(n()) # Standard Error of the Mean
  )

# Create the bar plot
bar_plot <- ggplot(plot_data, aes(x = reorder(Exercise, -mean_Loss), y = mean_Loss, fill = Exercise)) +
  geom_bar(stat = "identity", color = "black") +
  geom_errorbar(aes(ymin = mean_Loss - se_Loss, ymax = mean_Loss + se_Loss), width = 0.2) +
  labs(
    title = "Mean Loss by Exercise Group",
    x = "Exercise Type",
    y = "Mean Loss"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none" # Hide legend since the x-axis is self-explanatory
  )

# Save the plot as a high-quality PNG file
ggsave("Exercise_Loss_Barplot.png", plot = bar_plot, width = 8, height = 6)

# A friendly message confirming the file has been saved
cat("\n✅ Plot has been saved as 'Exercise_Loss_Barplot.png' in your working directory.\n")
