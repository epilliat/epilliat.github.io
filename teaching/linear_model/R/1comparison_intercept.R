# ================================================
# POLYNOMIAL REGRESSION AND OVERFITTING DEMONSTRATION
# ================================================


set.seed(42) # for reproducibility

# ----------------------------------------
# 1. GENERATE DATA
# ----------------------------------------

# %%

n <- 50 # number of observations

# True relationship: Y = 5 + 3*X + noise
# (simple linear relationship with some noise)
X <- runif(n, min = 0, max = 10)
Y <- 5 + 3 * X + rnorm(n, mean = 0, sd = 5)

# Create dataframe
data <- data.frame(X = X, Y = Y)
# ----------------------------------------
# 2. COMPARING MANUAL vs AUTOMATIC INTERCEPT
# ----------------------------------------

# %%

# Method 1: Manually adding intercept column
X_manual <- cbind(intercept = 1, X = X) # Add column of 1s for intercept

df_manual <- as.data.frame(X_manual)
model_manual <- lm(Y ~ intercept + X - 1, data = cbind(Y = Y, df_manual)) # -1 removes automatic intercept # nolint: line_length_linter.


summary(model_manual)

# %%

# Method 2: Automatic intercept
model_auto <- lm(Y ~ X, data = data)

summary(model_auto)

model_manual$coefficients == model_auto$coefficients
# %%
cat("\n=== COMPARISON ===\n")
cat("Manual intercept coefficient:", coef(model_manual)[1], "\n")
cat("Automatic intercept:", coef(model_auto)[1], "\n")
cat("Manual X coefficient:", coef(model_manual)[2], "\n")
cat("Automatic X coefficient:", coef(model_auto)[2], "\n")

cat("\n=== R-SQUARED COMPARISON ===\n")
cat("Manual model R²:", summary(model_manual)$r.squared, "\n")
cat("Automatic model R²:", summary(model_auto)$r.squared, "\n")
cat("Difference:", abs(summary(model_manual)$r.squared - summary(model_auto)$r.squared), "\n")

cat("\n=== ADJUSTED R-SQUARED COMPARISON ===\n")
cat("Manual model Adjusted R²:", summary(model_manual)$adj.r.squared, "\n")
cat("Automatic model Adjusted R²:", summary(model_auto)$adj.r.squared, "\n")
cat("Difference:", abs(summary(model_manual)$adj.r.squared - summary(model_auto)$adj.r.squared), "\n")

cat("\n=== RESIDUAL STANDARD ERROR COMPARISON ===\n")
cat("Manual model RSE:", summary(model_manual)$sigma, "\n")
cat("Automatic model RSE:", summary(model_auto)$sigma, "\n")

cat("\n=== F-STATISTIC COMPARISON ===\n")
cat("Manual model F-statistic:", summary(model_manual)$fstatistic[1], "\n")
cat("Automatic model F-statistic:", summary(model_auto)$fstatistic[1], "\n")


# %%
R2_m <- summary(model_manual)$r.squared
R2_a <- summary(model_auto)$r.squared

summary(model_manual)$fstatistic[1]
(R2_m) / (1 - (R2_m)) * (n - 2) / 2


summary(model_auto)$fstatistic[1]
(R2_a) / (1 - (R2_a)) * (n - 2) / 1

# Under the black box
