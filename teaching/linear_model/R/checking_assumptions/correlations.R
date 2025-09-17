# Autocorrelation Diagnostics in Linear Regression
# Testing for serial correlation in residuals

library(lmtest)
library(car)

set.seed(789)
n <- 200

# ==============================================================================
# GENERATE TIME SERIES DATA WITH DIFFERENT ERROR STRUCTURES
# ==============================================================================

# Time index
time <- 1:n

# Explanatory variables
X1 <- 0.5 * time + rnorm(n, 0, 5)
X2 <- sin(2 * pi * time / 50) * 10 + rnorm(n, 0, 2)
# Type "o" - overplotted points and lines (most common)
plot(X2,
    type = "o",
    main = 'type = "o" (points + lines)',
    xlab = "X", ylab = "Y",
    pch = 16, col = "blue", lwd = 2
)
# True parameters
beta0 <- 100
beta1 <- 2
beta2 <- 3

# Create three datasets with different error structures

# 1. NO AUTOCORRELATION (White noise errors)
errors_white <- rnorm(n, 0, 5)
Y_white <- beta0 + beta1 * X1 + beta2 * X2 + errors_white

# 2. POSITIVE AUTOCORRELATION (AR(1) with ρ = 0.7)
rho <- 0.7
errors_ar1 <- numeric(n)
errors_ar1[1] <- rnorm(1, 0, 5)
for (i in 2:n) {
    errors_ar1[i] <- rho * errors_ar1[i - 1] + rnorm(1, 0, 5 * sqrt(1 - rho^2))
}
Y_ar1 <- beta0 + beta1 * X1 + beta2 * X2 + errors_ar1

# 3. NEGATIVE AUTOCORRELATION (ρ = -0.6)
rho_neg <- -0.6
errors_ar1_neg <- numeric(n)
errors_ar1_neg[1] <- rnorm(1, 0, 5)
for (i in 2:n) {
    errors_ar1_neg[i] <- rho_neg * errors_ar1_neg[i - 1] + rnorm(1, 0, 5 * sqrt(1 - rho_neg^2))
}
Y_ar1_neg <- beta0 + beta1 * X1 + beta2 * X2 + errors_ar1_neg

# Create data frames
data_white <- data.frame(Y = Y_white, X1 = X1, X2 = X2, time = time)
data_ar1 <- data.frame(Y = Y_ar1, X1 = X1, X2 = X2, time = time)
data_ar1_neg <- data.frame(Y = Y_ar1_neg, X1 = X1, X2 = X2, time = time)

# ==============================================================================
# FIT MODELS
# ==============================================================================

model_white <- lm(Y ~ X1 + X2, data = data_white)
model_ar1 <- lm(Y ~ X1 + X2, data = data_ar1)
model_ar1_neg <- lm(Y ~ X1 + X2, data = data_ar1_neg)

# ==============================================================================
# VISUAL DIAGNOSTICS
# ==============================================================================

par(mfrow = c(3, 3))

# Plot residuals over time for each model
plot_residuals_time <- function(model, data, title, true_rho) {
    res <- residuals(model)

    # Residuals vs Time
    plot(data$time, res,
        main = paste(title, "- Residuals vs Time"),
        xlab = "Time", ylab = "Residuals",
        type = "o", pch = 16, col = rgb(0, 0, 1, 0.6)
    )
    abline(h = 0, col = "red", lwd = 2, lty = 2)

    # ACF plot
    acf(res, main = paste(title, "- ACF of Residuals"))

    # Residuals vs Lagged Residuals
    plot(res[-length(res)], res[-1],
        main = paste(title, "- εₜ vs εₜ₋₁"),
        xlab = "Residual(t-1)", ylab = "Residual(t)",
        pch = 16, col = rgb(0, 0, 1, 0.6)
    )
    abline(h = 0, v = 0, col = "gray", lty = 2)
    abline(lm(res[-1] ~ res[-length(res)]), col = "red", lwd = 2)

    # Add correlation coefficient
    cor_val <- cor(res[-length(res)], res[-1])
    legend("topleft",
        legend = paste("r =", round(cor_val, 3)),
        bty = "n", cex = 1.2
    )
}

plot_residuals_time(model_white, data_white, "No Autocorrelation", 0)
plot_residuals_time(model_ar1, data_ar1, "Positive Autocorrelation (ρ=0.7)", 0.7)
plot_residuals_time(model_ar1_neg, data_ar1_neg, "Negative Autocorrelation (ρ=-0.6)", -0.6)

par(mfrow = c(1, 1))

# ==============================================================================
# DURBIN-WATSON TEST
# ==============================================================================

cat("\n================== 2. DURBIN-WATSON TEST ==================\n\n")
cat("H₀: No first-order autocorrelation (ρ = 0)\n")
cat("DW statistic ≈ 2(1 - ρ), so:\n")
cat("  DW ≈ 2 → No autocorrelation\n")
cat("  DW < 2 → Positive autocorrelation\n")
cat("  DW > 2 → Negative autocorrelation\n\n")

# Durbin-Watson tests
dw_white <- dwtest(model_white)
dw_ar1 <- dwtest(model_ar1)
dw_ar1_neg <- dwtest(model_ar1_neg)


dw_white$statistic
dw_white$p.value

dw_ar1$statistic
dw_ar1$p.value

dw_ar1_neg$statistic
dw_ar1_neg$p.value

# ==============================================================================
# BREUSCH-GODFREY TEST
# ==============================================================================


# Test for different orders of autocorrelation
test_orders <- c(1, 2, 4)

for (order in test_orders) {
    cat(sprintf("Testing for AR(%d):\n", order))

    bg_white <- bgtest(model_white, order = order)
    bg_ar1 <- bgtest(model_ar1, order = order)
    bg_ar1_neg <- bgtest(model_ar1_neg, order = order)

    cat(sprintf(
        "  No Autocorr:    χ² = %6.2f, p-value = %.4f %s\n",
        bg_white$statistic, bg_white$p.value,
        ifelse(bg_white$p.value > 0.05, "✓", "✗")
    ))
    cat(sprintf(
        "  Positive AR(1): χ² = %6.2f, p-value = %.4f %s\n",
        bg_ar1$statistic, bg_ar1$p.value,
        ifelse(bg_ar1$p.value > 0.05, "✓", "✗")
    ))
    cat(sprintf(
        "  Negative AR(1): χ² = %6.2f, p-value = %.4f %s\n\n",
        bg_ar1_neg$statistic, bg_ar1_neg$p.value,
        ifelse(bg_ar1_neg$p.value > 0.05, "✓", "✗")
    ))
}

# ==============================================================================
# CONSEQUENCES OF AUTOCORRELATION
# ==============================================================================

cat("================== 4. CONSEQUENCES OF AUTOCORRELATION ==================\n\n")

# Compare standard errors: OLS vs Newey-West HAC
library(sandwich)

# Calculate robust standard errors for AR(1) model
se_ols <- sqrt(diag(vcov(model_ar1)))
se_nw <- sqrt(diag(NeweyWest(model_ar1)))

cat("Impact on Standard Errors (Positive Autocorrelation Model):\n")
cat("------------------------------------------------------------\n")
comparison <- data.frame(
    Coefficient = c("Intercept", "X1", "X2"),
    Estimate = coef(model_ar1),
    OLS_SE = se_ols,
    NeweyWest_SE = se_nw,
    SE_Ratio = se_nw / se_ols
)
print(comparison, digits = 4)

cat("\nNote: OLS standard errors are typically UNDERESTIMATED with positive\n")
cat("autocorrelation, leading to:\n")
cat("  • Overconfident inference (Type I errors)\n")
cat("  • Invalid confidence intervals\n")
cat("  • Misleading t-tests\n\n")

# ==============================================================================
# REMEDIES FOR AUTOCORRELATION
# ==============================================================================

cat("================== 5. REMEDIES FOR AUTOCORRELATION ==================\n\n")

cat("REMEDY 1: Cochrane-Orcutt Procedure\n")
cat("------------------------------------\n")

# Implement Cochrane-Orcutt
cochrane_orcutt <- function(model, data, max_iter = 10, tol = 0.001) {
    converged <- FALSE
    iter <- 0
    rho_old <- 0

    while (!converged && iter < max_iter) {
        if (iter == 0) {
            # Initial OLS
            res <- residuals(model)
        } else {
            # Transform variables
            y_trans <- data$Y[-1] - rho_old * data$Y[-n]
            x1_trans <- data$X1[-1] - rho_old * data$X1[-n]
            x2_trans <- data$X2[-1] - rho_old * data$X2[-n]

            model_trans <- lm(y_trans ~ x1_trans + x2_trans)
            res <- c(data$Y[1] - predict(model, newdata = data[1, ]), residuals(model_trans))
        }

        # Estimate rho
        rho_new <- cor(res[-length(res)], res[-1])

        # Check convergence
        if (abs(rho_new - rho_old) < tol) {
            converged <- TRUE
        }

        rho_old <- rho_new
        iter <- iter + 1
    }

    return(list(model = model_trans, rho = rho_new, iterations = iter))
}

# Apply Cochrane-Orcutt to positive autocorrelation model
co_result <- cochrane_orcutt(model_ar1, data_ar1)
cat(sprintf("Estimated ρ: %.3f (True ρ = 0.7)\n", co_result$rho))
cat(sprintf("Converged in %d iterations\n\n", co_result$iterations))

# Test the transformed model
dw_co <- dwtest(co_result$model)
cat(sprintf(
    "Durbin-Watson after Cochrane-Orcutt: %.3f (p = %.4f)\n",
    dw_co$statistic, dw_co$p.value
))
cat("Autocorrelation successfully removed!\n\n")

cat("REMEDY 2: Include Lagged Variables\n")
cat("-----------------------------------\n")
# Add lagged Y as predictor
data_ar1$Y_lag1 <- c(NA, data_ar1$Y[-n])
model_lagged <- lm(Y ~ X1 + X2 + Y_lag1, data = data_ar1[-1, ])
dw_lagged <- dwtest(model_lagged)
cat(sprintf(
    "Durbin-Watson with lagged Y: %.3f (p = %.4f)\n\n",
    dw_lagged$statistic, dw_lagged$p.value
))

cat("REMEDY 3: First Differencing\n")
cat("-----------------------------\n")
# First difference transformation
data_diff <- data.frame(
    Y_diff = diff(data_ar1$Y),
    X1_diff = diff(data_ar1$X1),
    X2_diff = diff(data_ar1$X2)
)
model_diff <- lm(Y_diff ~ X1_diff + X2_diff, data = data_diff)
dw_diff <- dwtest(model_diff)
cat(sprintf(
    "Durbin-Watson after differencing: %.3f (p = %.4f)\n\n",
    dw_diff$statistic, dw_diff$p.value
))

# ==============================================================================
# PRACTICAL GUIDELINES
# ==============================================================================

cat("================== PRACTICAL GUIDELINES ==================\n\n")

cat("WHEN TO WORRY ABOUT AUTOCORRELATION:\n")
cat("-------------------------------------\n")
cat("• Time series data (obvious temporal ordering)\n")
cat("• Panel/longitudinal data\n")
cat("• Spatial data (spatial autocorrelation)\n")
cat("• Any data with natural ordering\n\n")

cat("DETECTION STRATEGY:\n")
cat("-------------------\n")
cat("1. Plot residuals vs time/order\n")
cat("2. Check ACF/PACF plots\n")
cat("3. Run Durbin-Watson test (for AR(1))\n")
cat("4. Run Breusch-Godfrey test (for higher orders)\n\n")

cat("CHOOSING BETWEEN TESTS:\n")
cat("------------------------\n")
cat("Durbin-Watson:\n")
cat("  ✓ Simple and widely known\n")
cat("  ✓ Good for AR(1)\n")
cat("  ✗ Only tests first-order\n")
cat("  ✗ Requires specific conditions\n\n")

cat("Breusch-Godfrey:\n")
cat("  ✓ Tests any order of autocorrelation\n")
cat("  ✓ Works with lagged dependent variables\n")
cat("  ✓ More general and flexible\n")
cat("  ✗ Requires choosing the order to test\n\n")

cat("REMEDY SELECTION:\n")
cat("-----------------\n")
cat("• Known AR(1): Use Cochrane-Orcutt or Prais-Winsten\n")
cat("• General autocorrelation: Use HAC standard errors (Newey-West)\n")
cat("• Trending data: Consider first differencing\n")
cat("• Dynamic relationship: Add lagged variables\n")
cat("• Model misspecification: Check for omitted variables\n")
