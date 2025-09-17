# Linear Regression Linearity Assumption Diagnostics
# E(Y) = Xβ is the fundamental hypothesis

# Load required libraries
library(ggplot2)
library(gridExtra)

# Set seed for reproducibility
set.seed(123)

# Generate synthetic data with different relationships
n <- 200

# Create predictors
X1 <- runif(n, 0, 10)
X2 <- runif(n, 0, 10)
X3 <- runif(n, 0, 10)

# Create Y with:
# - Linear relationship with X1
# - Quadratic relationship with X2
# - Linear relationship with X3
# Plus some noise
Y <- 2 + 3 * X1 - 1 * X2^2 + 0.5 * X2 + 1.5 * X3 + rnorm(n, 0, 3)

# Combine into data frame
data <- data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3)

# ==============================================================================
# PRE-MODELING DIAGNOSTICS
# ==============================================================================

# 1. Scatter plots for each predictor
par(mfrow = c(2, 3))

# X1 vs Y (linear relationship)
plot(data$X1, data$Y,
    main = "Y vs X1 (Linear Relationship)",
    xlab = "X1", ylab = "Y",
    pch = 16, col = rgb(0, 0, 1, 0.5)
)
abline(lm(Y ~ X1, data = data), col = "red", lwd = 2)
grid()

# X2 vs Y (non-linear relationship)
plot(data$X2, data$Y,
    main = "Y vs X2 (Non-linear Relationship)",
    xlab = "X2", ylab = "Y",
    pch = 16, col = rgb(0, 0.5, 0, 0.5)
)
abline(lm(Y ~ X2, data = data), col = "red", lwd = 2)
grid()

# X3 vs Y (linear relationship)
plot(data$X3, data$Y,
    main = "Y vs X3 (Linear Relationship)",
    xlab = "X3", ylab = "Y",
    pch = 16, col = rgb(0.5, 0, 0.5, 0.5)
)
abline(lm(Y ~ X3, data = data), col = "red", lwd = 2)
grid()

# 2. Correlation analysis
cor(data)


# Fit the linear model
model <- lm(Y ~ X1 + X2 + X3, data = data)

# Model summary
summary(model)

# ==============================================================================
# POST-MODELING DIAGNOSTICS
# ==============================================================================

# Get residuals and fitted values
residuals <- residuals(model)
fitted_values <- fitted(model)

# 1. Residuals vs Fitted Values Plot
plot(fitted_values, residuals,
    main = "Residuals vs Fitted Values",
    xlab = "Fitted Values", ylab = "Residuals",
    pch = 16, col = rgb(0.2, 0.2, 0.2, 0.6)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)
# Add a loess smoother to highlight patterns
lines(lowess(fitted_values, residuals), col = "blue", lwd = 2)
grid()
legend("topright",
    legend = c("Zero line", "Loess smoother"),
    col = c("red", "blue"), lty = c(2, 1), lwd = 2
)

# 2. Residuals vs Individual Predictors
plot(data$X1, residuals,
    main = "Residuals vs X1",
    xlab = "X1", ylab = "Residuals",
    pch = 16, col = rgb(0, 0, 1, 0.5)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)
lines(lowess(data$X1, residuals), col = "blue", lwd = 2)
grid()

plot(data$X2, residuals,
    main = "Residuals vs X2 (Pattern indicates non-linearity)",
    xlab = "X2", ylab = "Residuals",
    pch = 16, col = rgb(0, 0.5, 0, 0.5)
)
abline(h = 0, col = "red", lwd = 2, lty = 2)
lines(lowess(data$X2, residuals), col = "blue", lwd = 2)
grid()

# Reset plot parameters
par(mfrow = c(1, 1))

# ==============================================================================
# INTERPRETATION
# ==============================================================================

cat("\n================== INTERPRETATION ==================\n\n")
cat("PRE-MODELING DIAGNOSTICS:\n")
cat("- Scatter plot of Y vs X1: Shows clear linear relationship\n")
cat("- Scatter plot of Y vs X2: Shows curvature (non-linear relationship)\n")
cat("- Scatter plot of Y vs X3: Shows linear relationship\n")
cat("- Correlation with X2 is weak due to non-linearity\n\n")

cat("POST-MODELING DIAGNOSTICS:\n")
cat("- Residuals vs Fitted: Shows some pattern, indicating model misspecification\n")
cat("- Residuals vs X1: Relatively random scatter (linearity assumption holds)\n")
cat("- Residuals vs X2: Clear U-shaped pattern (linearity assumption violated)\n")
cat("- Residuals vs X3: Relatively random scatter (linearity assumption holds)\n\n")

cat("CONCLUSION:\n")
cat("The non-linear relationship with X2 violates the linearity assumption.\n")
cat("Solutions could include:\n")
cat("1. Transform X2 (e.g., add X2^2 term)\n")
cat("2. Use non-linear regression methods\n")
cat("3. Apply splines or other flexible approaches\n\n")

# ==============================================================================
# DEMONSTRATION: Fixing the Non-linearity
# ==============================================================================

cat("================== FIXING THE NON-LINEARITY ==================\n\n")

# Add quadratic term for X2
data$X2_squared <- data$X2^2
model_fixed <- lm(Y ~ X1 + X2 + X2_squared + X3, data = data)

cat("Model with X2^2 term added:\n")
print(summary(model_fixed))

# Plot improved residuals
par(mfrow = c(1, 2))

# Original model residuals vs X2
plot(data$X2, residuals(model),
    main = "Original Model: Residuals vs X2",
    xlab = "X2", ylab = "Residuals",
    pch = 16, col = rgb(1, 0, 0, 0.5)
)
abline(h = 0, col = "black", lwd = 2, lty = 2)
lines(lowess(data$X2, residuals(model)), col = "blue", lwd = 2)

# Fixed model residuals vs X2
plot(data$X2, residuals(model_fixed),
    main = "Fixed Model: Residuals vs X2",
    xlab = "X2", ylab = "Residuals",
    pch = 16, col = rgb(0, 0.5, 0, 0.5)
)
abline(h = 0, col = "black", lwd = 2, lty = 2)
lines(lowess(data$X2, residuals(model_fixed)), col = "blue", lwd = 2)

par(mfrow = c(1, 1))

cat("\nThe fixed model shows much better residual patterns,\n")
cat("confirming that the linearity assumption now holds better.\n")
