# Understanding vcov(model) - The Variance-Covariance Matrix

library(sandwich)
set.seed(123)

# ==============================================================================
# WHAT IS vcov(model)?
# ==============================================================================

# vcov(model) returns the VARIANCE-COVARIANCE MATRIX of the regression coefficients
# It tells us:
# 1. The variance of each coefficient estimate (diagonal elements)
# 2. The covariance between coefficient estimates (off-diagonal elements)

# Simple example
n <- 100
X1 <- rnorm(n)
X2 <- rnorm(n)
Y <- 5 + 2 * X1 + 3 * X2 + rnorm(n)

model <- lm(Y ~ X1 + X2)

# Get the variance-covariance matrix
V <- vcov(model)
print(V)

# ==============================================================================
# UNDERSTANDING THE STRUCTURE
# ==============================================================================

# The matrix is symmetric and has dimensions (p x p) where p = number of coefficients
dim(V) # 3x3 matrix (intercept, X1, X2)

# DIAGONAL ELEMENTS = Variances of coefficients
# The variance of each coefficient estimate
var_intercept <- V[1, 1]
var_beta1 <- V[2, 2]
var_beta2 <- V[3, 3]

# Standard errors are the SQUARE ROOT of variances
se_intercept <- sqrt(V[1, 1])
se_beta1 <- sqrt(V[2, 2])
se_beta2 <- sqrt(V[3, 3])

# Verify: These match the standard errors from summary()
summary(model)$coefficients[, "Std. Error"]
c(se_intercept, se_beta1, se_beta2) # Same values!

# OFF-DIAGONAL ELEMENTS = Covariances between coefficients
cov_intercept_X1 <- V[1, 2] # Covariance between intercept and X1 coefficient
cov_intercept_X2 <- V[1, 3] # Covariance between intercept and X2 coefficient
cov_X1_X2 <- V[2, 3] # Covariance between X1 and X2 coefficients

# ==============================================================================
# WHY DO WE NEED vcov()?
# ==============================================================================

# 1. CALCULATE STANDARD ERRORS
# Standard errors are sqrt of diagonal elements
se_manual <- sqrt(diag(vcov(model)))
se_from_summary <- summary(model)$coefficients[, "Std. Error"]
all.equal(se_manual, se_from_summary) # TRUE

# 2. HYPOTHESIS TESTING
# t-statistics = coefficient / standard error
coefficients <- coef(model)
standard_errors <- sqrt(diag(vcov(model)))
t_statistics <- coefficients / standard_errors

# Compare with summary output
summary(model)$coefficients[, "t value"]
t_statistics # Same values!

# 3. CONFIDENCE INTERVALS
# 95% CI = coefficient ± 1.96 * SE
alpha <- 0.05
z_critical <- qnorm(1 - alpha / 2)

CI_lower <- coefficients - z_critical * standard_errors
CI_upper <- coefficients + z_critical * standard_errors

# Compare with confint()
confint(model, level = 0.95)
cbind(CI_lower, CI_upper) # Very similar (confint uses t-distribution)

# 4. LINEAR COMBINATIONS OF COEFFICIENTS
# What if we want to test β1 + β2 = 5?
# We need the variance of (β1 + β2)

# Variance of a linear combination: Var(aX + bY) = a²Var(X) + b²Var(Y) + 2ab*Cov(X,Y)
var_sum <- V[2, 2] + V[3, 3] + 2 * V[2, 3] # Var(β1 + β2)
se_sum <- sqrt(var_sum)

# Test if β1 + β2 = 5
sum_estimate <- coef(model)[2] + coef(model)[3]
z_stat <- (sum_estimate - 5) / se_sum
p_value <- 2 * pnorm(-abs(z_stat))

# ==============================================================================
# THE FORMULA BEHIND vcov()
# ==============================================================================

# vcov(model) = σ² * (X'X)^(-1)
# where σ² is the residual variance

# Manual calculation
X <- model.matrix(model) # Design matrix (includes intercept column)
sigma_squared <- sum(residuals(model)^2) / (n - 3) # Residual variance
vcov_manual <- sigma_squared * solve(t(X) %*% X)

# Compare with vcov()
vcov(model)
vcov_manual # Same matrix!

# ==============================================================================
# DIFFERENT TYPES OF vcov() - THE SANDWICH LIBRARY
# ==============================================================================

# When assumptions are violated, the standard vcov() is wrong
# The sandwich library provides corrected versions

# Create data with heteroscedasticity
X_hetero <- runif(100, 1, 10)
Y_hetero <- 5 + 2 * X_hetero + rnorm(100) * sqrt(X_hetero) # Variance increases with X
model_hetero <- lm(Y_hetero ~ X_hetero)

# STANDARD vcov (assumes homoscedasticity - WRONG here)
V_standard <- vcov(model_hetero)

# ROBUST vcov (corrects for heteroscedasticity - CORRECT)
V_robust <- vcovHC(model_hetero, type = "HC3")

# Compare standard errors
se_standard <- sqrt(diag(V_standard))
se_robust <- sqrt(diag(V_robust))

comparison <- data.frame(
    Standard_SE = se_standard,
    Robust_SE = se_robust,
    Ratio = se_robust / se_standard
)
print(comparison)

# Robust SEs are typically larger (more conservative) under heteroscedasticity

# ==============================================================================
# VISUALIZING THE VARIANCE-COVARIANCE MATRIX
# ==============================================================================

# The correlation between coefficient estimates
# Convert covariance to correlation
correlation_matrix <- cov2cor(vcov(model))

# Visualize as heatmap
library(corrplot)
corrplot(correlation_matrix,
    method = "number", type = "upper",
    title = "Correlation Between Coefficient Estimates"
)

# High correlation between coefficients can indicate multicollinearity

# ==============================================================================
# PRACTICAL USES OF vcov()
# ==============================================================================

# 1. TESTING MULTIPLE COEFFICIENTS SIMULTANEOUSLY (Wald test)
# Test if β1 = β2 = 0 (joint hypothesis)

# Subset of vcov for coefficients of interest
V_subset <- vcov(model)[2:3, 2:3] # Just X1 and X2

# Coefficient vector
beta_subset <- coef(model)[2:3]

# Wald statistic: β' * V^(-1) * β
wald_stat <- t(beta_subset) %*% solve(V_subset) %*% beta_subset
p_value_wald <- 1 - pchisq(wald_stat, df = 2)

# 2. PREDICTION INTERVALS
# Variance of prediction includes both parameter uncertainty and residual variance

new_data <- data.frame(X1 = 0.5, X2 = 1.0)
X_new <- model.matrix(~ X1 + X2, data = new_data)

# Variance of fitted value (parameter uncertainty only)
var_fit <- X_new %*% vcov(model) %*% t(X_new)
se_fit <- sqrt(var_fit)

# Prediction interval includes residual variance
var_pred <- var_fit + sigma(model)^2
se_pred <- sqrt(var_pred)

# Fitted value
fit <- predict(model, new_data)

# 95% Confidence interval for mean response
CI_fit <- c(fit - 1.96 * se_fit, fit + 1.96 * se_fit)

# 95% Prediction interval for individual response
PI <- c(fit - 1.96 * se_pred, fit + 1.96 * se_pred)

# Compare with predict() function
predict(model, new_data, interval = "confidence")
predict(model, new_data, interval = "prediction")

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================

# vcov(model) is fundamental for inference in regression:
# - Diagonal elements → Variances → Standard errors → t-tests
# - Off-diagonal elements → Covariances → Testing linear combinations
# - Different versions (HC, HAC, clustered) → Valid inference under violations
# - Foundation for confidence intervals, prediction intervals, hypothesis tests

# The formula: vcov = σ² * (X'X)^(-1)
# - Depends on residual variance (σ²)
# - Depends on design matrix (X)
# - Assumes homoscedasticity and independence (unless using robust versions)
