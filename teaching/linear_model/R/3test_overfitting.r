# %%
# ----------------------------------------
# 3. PROGRESSIVE POLYNOMIAL FITTING
# ----------------------------------------

cat("========================================\n")
cat("PART 2: POLYNOMIAL REGRESSION PROGRESSION\n")
cat("========================================\n\n")

# Storage for results
max_degree <- 50
results <- data.frame(
    Degree = 1:max_degree,
    R2 = numeric(max_degree),
    Adj_R2 = numeric(max_degree),
    AIC = numeric(max_degree),
    BIC = numeric(max_degree),
    RMSE = numeric(max_degree)
)

# Fit models of increasing polynomial degree
models_list <- list()

lm(Y~ X + X^2, data=data)

for (degree in 1:max_degree) {
    # Create polynomial formula
    if (degree == 1) {
        formula <- Y ~ X
    } else {
        poly_terms <- paste0("I(X^", 2:degree, ")", collapse = " + ")
        formula <- as.formula(paste("Y ~ X +", poly_terms))
    }

    # Fit model
    model <- lm(formula, data = data)
    models_list[[degree]] <- model

    # Store metrics
    results$R2[degree] <- summary(model)$r.squared
    results$Adj_R2[degree] <- summary(model)$adj.r.squared
    results$AIC[degree] <- AIC(model)
    results$BIC[degree] <- BIC(model)
    results$RMSE[degree] <- sqrt(mean(residuals(model)^2))
}

# Display results table
cat("=== MODEL COMPARISON TABLE ===\n")
print(results, digits = 4)

# ----------------------------------------
# 4. KEY OBSERVATIONS
# ----------------------------------------

cat("\n=== KEY OBSERVATIONS ===\n")
cat(
    "1. R² ALWAYS increases with more terms (",
    round(results$R2[1], 3), "→", round(results$R2[max_degree], 3), ")\n"
)
cat(
    "2. max Adjusted R² ",
    which.max(results$Adj_R2), ")\n"
)
cat("3. Best AIC at degree", which.min(results$AIC), "\n")
cat("4. Best BIC at degree", which.min(results$BIC), "\n")
cat("5. BIC penalizes complexity more than AIC\n\n")

# ----------------------------------------
# 5. VISUALIZATION
# ----------------------------------------

# Set up plotting area
par(mfrow = c(2, 3))

# Plot 1: Original data
plot(X, Y, main = "Original Data", pch = 19, col = "darkgray")
abline(lm(Y ~ X), col = "red", lwd = 2)

# Plot 2: Degree 3 (reasonable)
plot(X, Y, main = "Degree 3 Polynomial", pch = 19, col = "darkgray")
x_seq <- seq(min(X), max(X), length.out = 100)
pred_3 <- predict(models_list[[3]], newdata = data.frame(X = x_seq))
lines(x_seq, pred_3, col = "blue", lwd = 2)

# Plot 3: Degree 10 (overfitting)
plot(X, Y, main = "Degree 10 Polynomial (Overfitting)", pch = 19, col = "darkgray")
pred_10 <- predict(models_list[[10]], newdata = data.frame(X = x_seq))
lines(x_seq, pred_10, col = "purple", lwd = 2)

# Plot 4: R² vs Adjusted R²
plot(results$Degree, results$R2,
    type = "b", col = "red",
    ylim = c(min(results$Adj_R2), 1),
    xlab = "Polynomial Degree", ylab = "Value",
    main = "R² vs Adjusted R²", pch = 19
)
lines(results$Degree, results$Adj_R2, type = "b", col = "blue", pch = 17)
legend("bottomright",
    legend = c("R²", "Adjusted R²"),
    col = c("red", "blue"), pch = c(19, 17), lty = 1
)

# Plot 5: AIC and BIC
plot(results$Degree, results$AIC,
    type = "b", col = "green",
    xlab = "Polynomial Degree", ylab = "Value",
    main = "AIC vs BIC", pch = 19
)
lines(results$Degree, results$BIC, type = "b", col = "orange", pch = 17)
legend("topright",
    legend = c("AIC", "BIC"),
    col = c("green", "orange"), pch = c(19, 17), lty = 1
)

# Plot 6: RMSE
plot(results$Degree, results$RMSE,
    type = "b", col = "darkred",
    xlab = "Polynomial Degree", ylab = "RMSE",
    main = "Root Mean Square Error", pch = 19
)

par(mfrow = c(1, 1))

# ----------------------------------------
# 6. CROSS-VALIDATION DEMONSTRATION
# ----------------------------------------

cat("========================================\n")
cat("PART 3: CROSS-VALIDATION\n")
cat("========================================\n\n")

# Split data into training and test sets
set.seed(123)
train_idx <- sample(1:n, size = 0.7 * n)
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

# Calculate training and test errors
cv_results <- data.frame(
    Degree = 1:10,
    Train_RMSE = numeric(10),
    Test_RMSE = numeric(10)
)

for (degree in 1:10) {
    # Create formula
    if (degree == 1) {
        formula <- Y ~ X
    } else {
        poly_terms <- paste0("I(X^", 2:degree, ")", collapse = " + ")
        formula <- as.formula(paste("Y ~ X +", poly_terms))
    }

    # Fit on training data
    model <- lm(formula, data = train_data)

    # Predictions
    train_pred <- predict(model, newdata = train_data)
    test_pred <- predict(model, newdata = test_data)

    # Calculate RMSE
    cv_results$Train_RMSE[degree] <- sqrt(mean((train_data$Y - train_pred)^2))
    cv_results$Test_RMSE[degree] <- sqrt(mean((test_data$Y - test_pred)^2))
}

cat("=== TRAINING vs TEST ERROR ===\n")
print(cv_results, digits = 3)

# Plot training vs test error
plot(cv_results$Degree, cv_results$Train_RMSE,
    type = "b", col = "blue",
    ylim = c(min(cv_results$Train_RMSE), max(cv_results$Test_RMSE)),
    xlab = "Polynomial Degree", ylab = "RMSE",
    main = "Training vs Test Error (Overfitting Detection)", pch = 19
)
lines(cv_results$Degree, cv_results$Test_RMSE, type = "b", col = "red", pch = 17)
legend("topleft",
    legend = c("Training RMSE", "Test RMSE"),
    col = c("blue", "red"), pch = c(19, 17), lty = 1
)
abline(v = which.min(cv_results$Test_RMSE), lty = 2, col = "gray")
text(which.min(cv_results$Test_RMSE), max(cv_results$Test_RMSE),
    paste("Optimal degree:", which.min(cv_results$Test_RMSE)),
    pos = 4
)

# ----------------------------------------
# 7. FINAL SUMMARY
# ----------------------------------------

cat("\n========================================\n")
cat("SUMMARY AND RECOMMENDATIONS\n")
cat("========================================\n\n")

cat("1. INTERCEPT HANDLING:\n")
cat("   - R automatically includes intercept (no need for manual [1,X])\n")
cat("   - Use 'Y ~ X - 1' only if you want to force zero intercept\n\n")

cat("2. MODEL COMPLEXITY:\n")
cat("   - R² always increases with more parameters (not a good metric alone!)\n")
cat("   - Adjusted R² accounts for number of parameters\n")
cat("   - AIC/BIC balance fit and complexity\n\n")

cat("3. OVERFITTING SIGNS:\n")
cat("   - Adjusted R² decreasing while R² increases\n")
cat("   - Large gap between training and test error\n")
cat("   - Model fits noise rather than signal\n\n")

cat("4. BEST PRACTICES:\n")
cat("   - Use adjusted R², AIC, or BIC for model selection\n")
cat("   - Always validate on unseen data\n")
cat("   - Simpler models often generalize better\n")
cat("   - Consider the principle of parsimony (Occam's razor)\n")

results
# Display results table

poly_terms <- paste0("I(X^", 2:degree, ")", collapse = " + ")
formula <- as.formula(paste("Y ~ X +", poly_terms))
formula
poly_terms
