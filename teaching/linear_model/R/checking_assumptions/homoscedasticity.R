# Homoscedasticity Assumption Diagnostics in Linear Regression
# Var(ε|X) = σ² (constant variance assumption)

# Load required libraries
library(lmtest) # For Breusch-Pagan test
library(car) # Alternative BP test implementation

# Set seed for reproducibility
set.seed(456)

# ==============================================================================
# GENERATE SYNTHETIC DATA WITH HETEROSCEDASTICITY
# ==============================================================================

n <- 300

# Create predictors
X1 <- runif(n, 1, 20)
X2 <- runif(n, 1, 20)
X3 <- rnorm(n, 10, 3)

# Create two datasets:
# 1. Homoscedastic (constant variance)
# 2. Heteroscedastic (variance increases with fitted values)

# Dataset 1: Homoscedastic errors
Y_homo <- 5 + 2 * X1 + 1.5 * X2 - 0.8 * X3 + rnorm(n, 0, 3)

# Dataset 2: Heteroscedastic errors (variance increases with X1)
# Variance proportional to X1
heteroscedastic_errors <- rnorm(n, 0, 1) * (0.5 + 0.3 * X1)
Y_hetero <- 5 + 2 * X1 + 1.5 * X2 - 0.8 * X3 + heteroscedastic_errors

# Combine into data frames
data_homo <- data.frame(Y = Y_homo, X1 = X1, X2 = X2, X3 = X3)
data_hetero <- data.frame(Y = Y_hetero, X1 = X1, X2 = X2, X3 = X3)

# ==============================================================================
# FIT LINEAR MODELS
# ==============================================================================

model_homo <- lm(Y ~ X1 + X2 + X3, data = data_homo)
model_hetero <- lm(Y ~ X1 + X2 + X3, data = data_hetero)

# ==============================================================================
# VISUAL DIAGNOSTICS: RESIDUAL PLOTS
# ==============================================================================

par(mfrow = c(2, 3))

# --- HOMOSCEDASTIC MODEL ---
fitted_homo <- fitted(model_homo)
residuals_homo <- residuals(model_homo)
std_residuals_homo <- rstandard(model_homo)

# 1. Residuals vs Fitted
plot(fitted_homo, residuals_homo,
    main = "Homoscedastic Model: Residuals vs Fitted",
    xlab = "Fitted Values", ylab = "Residuals",
    pch = 16, col = rgb(0, 0, 1, 0.4)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)
# Add reference lines at ±2 standard deviations
abline(h = c(-2, 2) * sd(residuals_homo), col = "gray", lty = 3)

# 2. Scale-Location Plot (sqrt of standardized residuals)
plot(fitted_homo, sqrt(abs(std_residuals_homo)),
    main = "Homoscedastic Model: Scale-Location",
    xlab = "Fitted Values", ylab = "√|Standardized Residuals|",
    pch = 16, col = rgb(0, 0, 1, 0.4)
)
lines(lowess(fitted_homo, sqrt(abs(std_residuals_homo))), col = "red", lwd = 2)

# 3. Residuals vs X1 (main predictor)
plot(data_homo$X1, residuals_homo,
    main = "Homoscedastic Model: Residuals vs X1",
    xlab = "X1", ylab = "Residuals",
    pch = 16, col = rgb(0, 0, 1, 0.4)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)

# --- HETEROSCEDASTIC MODEL ---
fitted_hetero <- fitted(model_hetero)
residuals_hetero <- residuals(model_hetero)
std_residuals_hetero <- rstandard(model_hetero)

# 4. Residuals vs Fitted
plot(fitted_hetero, residuals_hetero,
    main = "Heteroscedastic Model: Residuals vs Fitted",
    xlab = "Fitted Values", ylab = "Residuals",
    pch = 16, col = rgb(1, 0, 0, 0.4)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)
# Add reference lines at ±2 standard deviations
abline(h = c(-2, 2) * sd(residuals_hetero), col = "gray", lty = 3)

# 5. Scale-Location Plot
plot(fitted_hetero, sqrt(abs(std_residuals_hetero)),
    main = "Heteroscedastic Model: Scale-Location",
    xlab = "Fitted Values", ylab = "√|Standardized Residuals|",
    pch = 16, col = rgb(1, 0, 0, 0.4)
)
lines(lowess(fitted_hetero, sqrt(abs(std_residuals_hetero))), col = "red", lwd = 2)

# 6. Residuals vs X1
plot(data_hetero$X1, residuals_hetero,
    main = "Heteroscedastic Model: Residuals vs X1",
    xlab = "X1", ylab = "Residuals",
    pch = 16, col = rgb(1, 0, 0, 0.4)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)

par(mfrow = c(1, 1))

# ==============================================================================
# BREUSCH-PAGAN TEST
# ==============================================================================

cat("\n================== BREUSCH-PAGAN TEST RESULTS ==================\n\n")

# Test for homoscedastic model
bp_homo <- bptest(model_homo)
# Test for heteroscedastic model
bp_hetero <- bptest(model_hetero)

bp_homo$statistic
bp_homo$p.value
bp_hetero$statistic
bp_hetero$p.value
# ==============================================================================
# ADDITIONAL DIAGNOSTIC: WHITE'S TEST (more general)
# ==============================================================================

cat("\n================== ADDITIONAL TESTS ==================\n\n")

# Using ncvTest from car package (similar to White's test)
ncv_homo <- ncvTest(model_homo)
ncv_hetero <- ncvTest(model_hetero)

ncv_homo$ChiSquare
ncv_homo$p
ncv_hetero$ChiSquare
ncv_hetero$p

# ==============================================================================
# VARIANCE ANALYSIS BY FITTED VALUE GROUPS
# ==============================================================================

cat("\n================== VARIANCE BY FITTED VALUE GROUPS ==================\n\n")

# Divide fitted values into quartiles and calculate variance in each group
analyze_variance_by_group <- function(fitted, residuals, model_name) {
    quartiles <- quantile(fitted, probs = c(0, 0.25, 0.5, 0.75, 1))
    groups <- cut(fitted, breaks = quartiles, include.lowest = TRUE)

    cat(sprintf("%s:\n", model_name))
    cat("Residual variance by fitted value quartile:\n")

    variances <- tapply(residuals, groups, var)
    for (i in 1:length(variances)) {
        cat(sprintf("  Q%d: %.3f\n", i, variances[i]))
    }

    ratio <- max(variances) / min(variances)
    cat(sprintf("  Ratio (max/min): %.2f\n", ratio))
    if (ratio > 3) {
        cat("  Warning: Large variance ratio suggests heteroscedasticity\n")
    }
    cat("\n")
}

analyze_variance_by_group(fitted_homo, residuals_homo, "Homoscedastic Model")
analyze_variance_by_group(fitted_hetero, residuals_hetero, "Heteroscedastic Model")

# ==============================================================================
# CONSEQUENCES & REMEDIES
# ==============================================================================

cat("================== INTERPRETATION & REMEDIES ==================\n\n")

cat("VISUAL DIAGNOSTICS INTERPRETATION:\n")
cat("-----------------------------------\n")
cat("Homoscedastic Model:\n")
cat("  - Residuals show constant spread across fitted values\n")
cat("  - Scale-Location plot shows horizontal trend\n")
cat("  - Random scatter pattern in all residual plots\n\n")

cat("Heteroscedastic Model:\n")
cat("  - Residuals show 'funnel' or 'cone' shape\n")
cat("  - Scale-Location plot shows upward trend\n")
cat("  - Variance clearly increases with fitted values\n\n")

cat("CONSEQUENCES OF HETEROSCEDASTICITY:\n")
cat("------------------------------------\n")
cat("1. OLS estimates remain unbiased but inefficient\n")
cat("2. Standard errors are biased (usually underestimated)\n")
cat("3. Confidence intervals and hypothesis tests are invalid\n")
cat("4. Predictions have varying reliability\n\n")

cat("REMEDIES FOR HETEROSCEDASTICITY:\n")
cat("---------------------------------\n")
cat("1. Transform the response variable (log, sqrt)\n")
cat("2. Use Weighted Least Squares (WLS)\n")
cat("3. Use robust standard errors (HC0, HC1, HC2, HC3)\n")
cat("4. Model the variance explicitly (e.g., GARCH models)\n\n")
