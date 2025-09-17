# Leverage, Outliers, and Cook's Distance in Regression Diagnostics

library(car)
set.seed(123)

# ==============================================================================
# CREATE DEMONSTRATION DATASET
# ==============================================================================

n <- 50
X <- runif(n, 0, 10)
Y <- 2 + 1.5 * X + rnorm(n, 0, 1)

# Add different types of problematic points
# 1. High leverage, not outlier (follows pattern but extreme X)
X <- c(X, 15)
Y <- c(Y, 2 + 1.5 * 15 + 0.5) # On regression line

# 2. Outlier, low leverage (unusual Y, typical X)
X <- c(X, 5)
Y <- c(Y, 15) # Far from regression line

# 3. High leverage AND outlier (influential point)
X <- c(X, 14)
Y <- c(Y, 5) # Extreme X and doesn't follow pattern

# 4. Another influential point
X <- c(X, -2)
Y <- c(Y, 12)

# Label points
point_types <- c(
    rep("Normal", n),
    "High Leverage Only",
    "Outlier Only",
    "Influential (Both)",
    "Influential (Both)"
)

data <- data.frame(X = X, Y = Y, Type = point_types)

# ==============================================================================
# FIRST: VISUALIZE THE RAW DATA
# ==============================================================================

par(mfrow = c(2, 2))

# Plot 1: Raw data with all points
plot(X, Y,
    main = "Raw Data - All Points",
    xlab = "X", ylab = "Y",
    pch = 19, col = rgb(0, 0, 1, 0.6),
    xlim = c(min(X) - 1, max(X) + 1),
    ylim = c(min(Y) - 1, max(Y) + 1)
)
grid()

# Highlight special points
special_idx <- which(point_types != "Normal")
points(X[special_idx], Y[special_idx], pch = 19, col = "red", cex = 1.5)

# Add point labels
text(X[special_idx], Y[special_idx],
    labels = paste0("#", special_idx, ": ", point_types[special_idx]),
    pos = c(4, 4, 2, 4), cex = 0.7, col = "darkred"
)

# Plot 2: Data with regression line (all points)
plot(X, Y,
    main = "With Regression Line (All Points)",
    xlab = "X", ylab = "Y",
    pch = 19, col = rgb(0, 0, 1, 0.6),
    xlim = c(min(X) - 1, max(X) + 1),
    ylim = c(min(Y) - 1, max(Y) + 1)
)
grid()

# Fit and add regression line
model_all <- lm(Y ~ X, data = data)
abline(model_all, col = "blue", lwd = 2)

# Highlight special points
points(X[special_idx], Y[special_idx], pch = 19, col = "red", cex = 1.5)

# Add equation
eq_all <- sprintf("y = %.2f + %.2fx", coef(model_all)[1], coef(model_all)[2])
legend("topleft", legend = eq_all, bty = "n", text.col = "blue")

# Plot 3: Data without special points
normal_idx <- which(point_types == "Normal")
plot(X[normal_idx], Y[normal_idx],
    main = "Without Special Points (Normal Only)",
    xlab = "X", ylab = "Y",
    pch = 19, col = rgb(0, 0.7, 0, 0.6),
    xlim = c(min(X) - 1, max(X) + 1),
    ylim = c(min(Y) - 1, max(Y) + 1)
)
grid()

# Fit and add regression line for normal points only
model_normal <- lm(Y ~ X, data = data[normal_idx, ])
abline(model_normal, col = "darkgreen", lwd = 2)

# Add equation
eq_normal <- sprintf("y = %.2f + %.2fx", coef(model_normal)[1], coef(model_normal)[2])
legend("topleft", legend = eq_normal, bty = "n", text.col = "darkgreen")

# Plot 4: Comparison of both regression lines
plot(X, Y,
    main = "Comparison: Effect of Special Points",
    xlab = "X", ylab = "Y",
    pch = 19, col = ifelse(point_types == "Normal", rgb(0, 0, 1, 0.4), rgb(1, 0, 0, 0.8)),
    xlim = c(min(X) - 1, max(X) + 1),
    ylim = c(min(Y) - 1, max(Y) + 1)
)
grid()

# Add both regression lines
abline(model_all, col = "blue", lwd = 2, lty = 1)
abline(model_normal, col = "darkgreen", lwd = 2, lty = 2)

# Add legend
legend("topleft",
    legend = c("With all points", "Normal points only"),
    col = c("blue", "darkgreen"),
    lty = c(1, 2), lwd = 2, bty = "n"
)

# Label special points
text(X[special_idx], Y[special_idx], labels = special_idx, pos = 3, cex = 0.8)

par(mfrow = c(1, 1))

# ==============================================================================
# UNDERSTANDING THE CONCEPTS
# ==============================================================================

# LEVERAGE: How far an observation's X values are from the mean of X
#          High leverage = unusual X values
#          Formula: h_ii = diagonal of Hat matrix H = X(X'X)^(-1)X'

# OUTLIER: Observation with large residual (unusual Y given X)
#         Measured by standardized or studentized residuals

# INFLUENCE: Combined effect of leverage AND outlier status
#           High influence = changes regression substantially if removed
#           Measured by Cook's Distance

# ==============================================================================
# CALCULATE DIAGNOSTIC MEASURES
# ==============================================================================

# Fit model with all data
model <- lm(Y ~ X, data = data)

# 1. LEVERAGE (Hat values)
leverage <- hatvalues(model)
p <- length(coef(model))
n_obs <- nrow(data)
leverage_threshold <- 2 * p / n_obs

# 2. STANDARDIZED RESIDUALS
std_residuals <- rstandard(model)

# 3. STUDENTIZED RESIDUALS
stud_residuals <- rstudent(model)

# 4. COOK'S DISTANCE
cooks_d <- cooks.distance(model)

# Create diagnostic summary
diagnostics <- data.frame(
    Observation = 1:length(X),
    Type = point_types,
    X = X,
    Y = Y,
    Leverage = round(leverage, 3),
    High_Leverage = leverage > leverage_threshold,
    Std_Residual = round(std_residuals, 2),
    Stud_Residual = round(stud_residuals, 2),
    Outlier = abs(stud_residuals) > 2,
    Cooks_D = round(cooks_d, 4),
    Influential = cooks_d > 4 / n_obs
)

# Display summary of special points
cat("================== DATA SUMMARY ==================\n\n")
cat(sprintf("Total observations: %d\n", n_obs))
cat(sprintf("Normal points: %d\n", sum(point_types == "Normal")))
cat(sprintf("Special points: %d\n\n", sum(point_types != "Normal")))

cat("Special points added:\n")
cat("1. Point #51: High Leverage Only (X=15, Y≈expected)\n")
cat("2. Point #52: Outlier Only (X=5, Y=15)\n")
cat("3. Point #53: Influential (X=14, Y=5)\n")
cat("4. Point #54: Influential (X=-2, Y=12)\n\n")

# Display key diagnostic measures
cat("================== DIAGNOSTIC MEASURES FOR SPECIAL POINTS ==================\n\n")
key_points <- diagnostics[diagnostics$Type != "Normal", ]
print(key_points[, c("Observation", "Type", "X", "Y", "Leverage", "Stud_Residual", "Cooks_D")])

cat("\nThresholds used:\n")
cat(sprintf("  High leverage: h_ii > %.3f (2p/n = 2*%d/%d)\n", leverage_threshold, p, n_obs))
cat(sprintf("  Outlier: |Studentized Residual| > 2\n"))
cat(sprintf("  Influential: Cook's D > %.3f (4/n)\n\n", 4 / n_obs))

# ==============================================================================
# DIAGNOSTIC PLOTS
# ==============================================================================

par(mfrow = c(2, 3))

# Plot 1: Leverage values
plot(1:n_obs, leverage,
    type = "h",
    main = "Leverage Values (Hat Values)",
    xlab = "Observation Index", ylab = "Leverage",
    ylim = c(0, max(leverage) * 1.1)
)
points(1:n_obs, leverage,
    pch = 19,
    col = ifelse(leverage > leverage_threshold, "red", "blue")
)
abline(h = leverage_threshold, col = "red", lty = 2)
text(
    x = n_obs * 0.9, y = leverage_threshold * 1.1,
    labels = sprintf("Threshold = %.3f", leverage_threshold), col = "red", cex = 0.8
)

# Label high leverage points
high_lev_idx <- which(leverage > leverage_threshold)
text(high_lev_idx, leverage[high_lev_idx], labels = high_lev_idx, pos = 3, cex = 0.8)

# Plot 2: Studentized residuals
plot(1:n_obs, stud_residuals,
    type = "h",
    main = "Studentized Residuals",
    xlab = "Observation Index", ylab = "Studentized Residuals",
    ylim = range(stud_residuals) * 1.1
)
points(1:n_obs, stud_residuals,
    pch = 19,
    col = ifelse(abs(stud_residuals) > 2, "red", "blue")
)
abline(h = c(-2, 0, 2), col = c("red", "gray", "red"), lty = c(2, 1, 2))

# Label outliers
outlier_idx <- which(abs(stud_residuals) > 2)
text(outlier_idx, stud_residuals[outlier_idx], labels = outlier_idx, pos = 3, cex = 0.8)

# Plot 3: Cook's Distance
plot(1:n_obs, cooks_d,
    type = "h",
    main = "Cook's Distance",
    xlab = "Observation Index", ylab = "Cook's D",
    ylim = c(0, max(cooks_d) * 1.1)
)
points(1:n_obs, cooks_d,
    pch = 19,
    col = ifelse(cooks_d > 4 / n_obs, "red", "blue")
)
abline(h = 4 / n_obs, col = "red", lty = 2)
abline(h = 1, col = "orange", lty = 2)

# Label thresholds and influential points
text(n_obs * 0.9, 4 / n_obs * 1.1, "4/n threshold", col = "red", cex = 0.8)
influential_idx <- which(cooks_d > 4 / n_obs)
text(influential_idx, cooks_d[influential_idx], labels = influential_idx, pos = 3, cex = 0.8)

# Plot 4: Residuals vs Leverage
plot(leverage, stud_residuals,
    main = "Residuals vs Leverage",
    xlab = "Leverage", ylab = "Studentized Residuals",
    pch = 19, col = ifelse(cooks_d > 4 / n_obs, "red", "blue")
)
abline(h = c(-2, 0, 2), col = "gray", lty = 2)
abline(v = leverage_threshold, col = "gray", lty = 2)

# Add Cook's distance contours
x_lev <- seq(0, max(leverage), length.out = 100)
cook_levels <- c(4 / n_obs, 0.5)
for (cook_level in cook_levels) {
    y_pos <- sqrt(cook_level * p * (1 - x_lev) / x_lev)
    y_neg <- -y_pos
    lines(x_lev, y_pos, col = "darkgreen", lty = 3)
    lines(x_lev, y_neg, col = "darkgreen", lty = 3)
}

# Label points
text(leverage[special_idx], stud_residuals[special_idx],
    labels = special_idx, pos = 3, cex = 0.8
)

# Add quadrant labels
text(leverage_threshold / 2, 3, "Outliers\n(Low leverage)", cex = 0.7, col = "gray")
text(max(leverage) * 0.8, 0, "High Leverage\n(Good fit)", cex = 0.7, col = "gray")
text(max(leverage) * 0.8, max(stud_residuals) * 0.8, "Influential\n(Both)", cex = 0.7, col = "darkred")

# Plot 5: Influence Plot (Bubble plot)
plot(leverage, stud_residuals,
    main = "Influence Plot (Bubble size = Cook's D)",
    xlab = "Leverage", ylab = "Studentized Residuals",
    type = "n"
)
abline(h = c(-2, 0, 2), col = "gray", lty = 2)
abline(v = leverage_threshold, col = "gray", lty = 2)

# Add bubbles proportional to Cook's D
symbols(leverage, stud_residuals,
    circles = sqrt(cooks_d),
    inches = 0.3, fg = "blue", bg = rgb(0, 0, 1, 0.3), add = TRUE
)
text(leverage[special_idx], stud_residuals[special_idx],
    labels = special_idx, cex = 0.7
)

# Plot 6: DFBETAS for slope
dfbetas_vals <- dfbetas(model)
plot(1:n_obs, dfbetas_vals[, 2],
    type = "h",
    main = "DFBETAS for Slope (Impact on β₁)",
    xlab = "Observation Index", ylab = "DFBETAS",
    ylim = range(dfbetas_vals[, 2]) * 1.1
)
points(1:n_obs, dfbetas_vals[, 2],
    pch = 19,
    col = ifelse(abs(dfbetas_vals[, 2]) > 2 / sqrt(n_obs), "red", "blue")
)
abline(h = c(-2 / sqrt(n_obs), 0, 2 / sqrt(n_obs)), col = c("red", "gray", "red"), lty = c(2, 1, 2))

high_dfbeta_idx <- which(abs(dfbetas_vals[, 2]) > 2 / sqrt(n_obs))
text(high_dfbeta_idx, dfbetas_vals[high_dfbeta_idx, 2],
    labels = high_dfbeta_idx, pos = 3, cex = 0.8
)

par(mfrow = c(1, 1))

# ==============================================================================
# COMPARING MODELS WITH AND WITHOUT INFLUENTIAL POINTS
# =======================================
