# Normality Testing of Residuals in Linear Regression
# QQ-Plot, Shapiro-Wilk Test, and Kolmogorov-Smirnov Test

library(car) # For additional diagnostic plots
set.seed(456)
n <- 200

# ==============================================================================
# CREATE DATASETS WITH DIFFERENT ERROR DISTRIBUTIONS
# ==============================================================================

# Common design for all models
X1 <- runif(n, 1, 20)
X2 <- rnorm(n, 10, 3)
beta0 <- 5
beta1 <- 2
beta2 <- 1.5

# Dataset 1: NORMAL ERRORS (assumption satisfied)
errors_normal <- rnorm(n, 0, 3)
Y_normal <- beta0 + beta1 * X1 + beta2 * X2 + errors_normal

# Dataset 2: HEAVY-TAILED ERRORS (t-distribution)
errors_heavy <- rt(n, df = 3) * 3 # t-distribution with 3 df
Y_heavy <- beta0 + beta1 * X1 + beta2 * X2 + errors_heavy

# Dataset 3: SKEWED ERRORS (exponential)
errors_skewed <- (rexp(n, rate = 0.5) - 2) * 2 # Shifted and scaled exponential
Y_skewed <- beta0 + beta1 * X1 + beta2 * X2 + errors_skewed

# Dataset 4: BIMODAL ERRORS (mixture of normals)
# 70% from N(0, 2) and 30% from N(8, 2)
mixture_indicator <- rbinom(n, 1, 0.3)
errors_bimodal <- ifelse(mixture_indicator == 0, rnorm(n, 0, 2), rnorm(n, 8, 2))
Y_bimodal <- beta0 + beta1 * X1 + beta2 * X2 + errors_bimodal

# Fit models
model_normal <- lm(Y_normal ~ X1 + X2)
model_heavy <- lm(Y_heavy ~ X1 + X2)
model_skewed <- lm(Y_skewed ~ X1 + X2)
model_bimodal <- lm(Y_bimodal ~ X1 + X2)

# ==============================================================================
# UNDERSTANDING QQ-PLOTS
# ==============================================================================

# QQ-plot compares sample quantiles to theoretical normal quantiles
# Points should fall approximately on a straight line if normal

par(mfrow = c(2, 2))

# Function to create enhanced QQ-plot
create_qq_plot <- function(residuals, title, subtitle) {
  qqnorm(residuals,
    main = title,
    sub = subtitle,
    pch = 19, col = rgb(0, 0, 1, 0.5),
    xlab = "Theoretical Quantiles",
    ylab = "Sample Quantiles"
  )
  qqline(residuals, col = "red", lwd = 2)

  # Add confidence bands (approximate)
  n <- length(residuals)
  sorted_residuals <- sort(residuals)
  theoretical_quantiles <- qnorm(ppoints(n))

  # Simple confidence bands (rough approximation)
  se <- sd(residuals) * sqrt(pnorm(theoretical_quantiles) * (1 - pnorm(theoretical_quantiles)) / n)
  # lines(theoretical_quantiles, theoretical_quantiles * sd(residuals) + mean(residuals) + 2*se,
  #       col = "gray", lty = 2)
  # lines(theoretical_quantiles, theoretical_quantiles * sd(residuals) + mean(residuals) - 2*se,
  #       col = "gray", lty = 2)
}

# Create QQ-plots for all models
create_qq_plot(residuals(model_normal), "Normal Residuals", "Points follow line")
create_qq_plot(residuals(model_heavy), "Heavy-Tailed Residuals", "S-shaped: tails diverge")
create_qq_plot(residuals(model_skewed), "Skewed Residuals", "Curved: systematic deviation")
create_qq_plot(residuals(model_bimodal), "Bimodal Residuals", "Multiple patterns")

# Histogram comparison
hist(residuals(model_normal), main = "Normal", col = "lightblue", breaks = 20)
hist(residuals(model_heavy), main = "Heavy-Tailed", col = "lightcoral", breaks = 20)
hist(residuals(model_skewed), main = "Skewed", col = "lightgreen", breaks = 20)
hist(residuals(model_bimodal), main = "Bimodal", col = "lightyellow", breaks = 20)

par(mfrow = c(1, 1))

# ==============================================================================
# SHAPIRO-WILK TEST
# ==============================================================================

# H₀: Residuals are normally distributed
# H₁: Residuals are not normally distributed
# p-value < 0.05 → Reject normality

# Shapiro-Wilk has a sample size limit (3 ≤ n ≤ 5000)

# Perform Shapiro-Wilk tests
sw_normal <- shapiro.test(residuals(model_normal))
sw_heavy <- shapiro.test(residuals(model_heavy))
sw_skewed <- shapiro.test(residuals(model_skewed))
sw_bimodal <- shapiro.test(residuals(model_bimodal))

sw_normal
sw_heavy
sw_skewed
sw_bimodal
# Display results

shapiro_results <- data.frame(
  Model = c("Normal", "Heavy-Tailed", "Skewed", "Bimodal"),
  W_Statistic = c(
    sw_normal$statistic, sw_heavy$statistic,
    sw_skewed$statistic, sw_bimodal$statistic
  ),
  p_value = c(
    sw_normal$p.value, sw_heavy$p.value,
    sw_skewed$p.value, sw_bimodal$p.value
  ),
  Conclusion = c(
    ifelse(sw_normal$p.value > 0.05, "Normal ✓", "Not Normal ✗"),
    ifelse(sw_heavy$p.value > 0.05, "Normal ✓", "Not Normal ✗"),
    ifelse(sw_skewed$p.value > 0.05, "Normal ✓", "Not Normal ✗"),
    ifelse(sw_bimodal$p.value > 0.05, "Normal ✓", "Not Normal ✗")
  )
)

print(shapiro_results)

# ==============================================================================
# KOLMOGOROV-SMIRNOV TEST
# ==============================================================================

# KS test compares the empirical CDF with theoretical normal CDF
# Need to standardize residuals first

# Standardize residuals
standardize <- function(x) (x - mean(x)) / sd(x)

# Perform KS tests
ks_normal <- ks.test(standardize(residuals(model_normal)), "pnorm")
ks_heavy <- ks.test(standardize(residuals(model_heavy)), "pnorm")
ks_skewed <- ks.test(standardize(residuals(model_skewed)), "pnorm")
ks_bimodal <- ks.test(standardize(residuals(model_bimodal)), "pnorm")


ks_results <- data.frame(
  Model = c("Normal", "Heavy-Tailed", "Skewed", "Bimodal"),
  D_Statistic = c(
    ks_normal$statistic, ks_heavy$statistic,
    ks_skewed$statistic, ks_bimodal$statistic
  ),
  p_value = c(
    ks_normal$p.value, ks_heavy$p.value,
    ks_skewed$p.value, ks_bimodal$p.value
  ),
  Conclusion = c(
    ifelse(ks_normal$p.value > 0.05, "Normal ✓", "Not Normal ✗"),
    ifelse(ks_heavy$p.value > 0.05, "Normal ✓", "Not Normal ✗"),
    ifelse(ks_skewed$p.value > 0.05, "Normal ✓", "Not Normal ✗"),
    ifelse(ks_bimodal$p.value > 0.05, "Normal ✓", "Not Normal ✗")
  )
)

print(ks_results)

# ==============================================================================
# COMPARING TESTS: POWER AND CHARACTERISTICS
# ==============================================================================

comparison <- data.frame(
  Model = c("Normal", "Heavy-Tailed", "Skewed", "Bimodal"),
  Shapiro_p = round(c(
    sw_normal$p.value, sw_heavy$p.value,
    sw_skewed$p.value, sw_bimodal$p.value
  ), 4),
  KS_p = round(c(
    ks_normal$p.value, ks_heavy$p.value,
    ks_skewed$p.value, ks_bimodal$p.value
  ), 4),
  Both_Agree = c(
    (sw_normal$p.value > 0.05) == (ks_normal$p.value > 0.05),
    (sw_heavy$p.value > 0.05) == (ks_heavy$p.value > 0.05),
    (sw_skewed$p.value > 0.05) == (ks_skewed$p.value > 0.05),
    (sw_bimodal$p.value > 0.05) == (ks_bimodal$p.value > 0.05)
  )
)

print(comparison)
