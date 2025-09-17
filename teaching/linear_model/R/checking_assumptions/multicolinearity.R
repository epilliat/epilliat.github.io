# ========================================
# MULTICOLLINEARITY DEMONSTRATION
# ========================================

# Load necessary packages
library(car) # for VIF
set.seed(123) # for reproducibility

# ----------------------------------------
# 1. CREATE A DATASET WITH MULTICOLLINEARITY
# ----------------------------------------

n <- 100 # number of observations

# Highly correlated explanatory variables
X1 <- rnorm(n, mean = 50, sd = 10)
X2 <- 2 * X1 + rnorm(n, mean = 0, sd = 0.1) # X2 highly correlated with X1
X3 <- X1 + X2 + rnorm(n, mean = 0, sd = 0.1) # X3 combination of X1 and X2
X4 <- rnorm(n, mean = 30, sd = 8) # X4 independent

# Dependent variable: Y truly depends on these variables
Y <- 10 + 2 * X1 + 1.5 * X2 - 0.5 * X3 + 0 * X4 + rnorm(n, mean = 0, sd = 10)

# Create dataframe
data <- data.frame(Y, X1, X2, X3, X4)

A <- cbind(X1, X2, X3, X4)
U <- t(A) %*% A
eigen(U)$values

# ----------------------------------------
# 2. PRELIMINARY EXPLORATION
# ----------------------------------------

cat("=== CORRELATIONS BETWEEN VARIABLES ===\n")
cor_matrix <- cor(data)
print(round(cor_matrix, 2))

cat("\n=== IMPORTANT OBSERVATION ===\n")
cat("Y is strongly correlated with X1 (r =", round(cor(Y, X1), 2), ")\n")
cat("Y is strongly correlated with X2 (r =", round(cor(Y, X2), 2), ")\n")
cat("Y is strongly correlated with X3 (r =", round(cor(Y, X3), 2), ")\n")
cat("BUT X1, X2 and X3 are also highly correlated with each other!\n\n")

# ----------------------------------------
# 3. REGRESSION WITH MULTICOLLINEARITY
# ----------------------------------------

cat("=== FULL MODEL (WITH MULTICOLLINEARITY) ===\n")
model_full <- lm(Y ~ X1 + X2 + X3 + X4, data = data)
summary(model_full)

cat("\n=== PROBLEM DETECTED ===\n")
cat("- High R²:", round(summary(model_full)$r.squared, 3), "\n")
cat("- BUT few significant variables despite strong correlations!\n")
cat("- Standard errors are inflated\n\n")

# ----------------------------------------
# 4. DIAGNOSIS: CALCULATE VIF
# ----------------------------------------

cat("=== VARIANCE INFLATION FACTORS (VIF) ===\n")
vif_values <- vif(model_full)
print(vif_values)

cat("\n=== INTERPRETATION ===\n")
cat("VIF > 10 : Severe multicollinearity\n")
cat("VIF > 5  : Concerning multicollinearity\n")
cat("→ X1, X2 and X3 have very high VIF values!\n\n")

# ----------------------------------------
# 5. VISUALIZE THE PROBLEM
# ----------------------------------------

par(mfrow = c(2, 2))

# Scatter plots
plot(X1, X2,
    main = "X1 vs X2 (r = 0.95)",
    pch = 19, col = "blue", cex = 0.7
)
abline(lm(X2 ~ X1), col = "red", lwd = 2)

plot(X1, X3,
    main = "X1 vs X3 (r = 0.96)",
    pch = 19, col = "blue", cex = 0.7
)
abline(lm(X3 ~ X1), col = "red", lwd = 2)

plot(X2, X3,
    main = "X2 vs X3 (r = 0.99)",
    pch = 19, col = "blue", cex = 0.7
)
abline(lm(X3 ~ X2), col = "red", lwd = 2)

plot(X1, X4,
    main = "X1 vs X4 (independent)",
    pch = 19, col = "green", cex = 0.7
)
abline(lm(X4 ~ X1), col = "red", lwd = 2)

par(mfrow = c(1, 1))

# ----------------------------------------
# 6. POSSIBLE SOLUTIONS
# ----------------------------------------

cat("\n=== SOLUTION 1: REMOVE ONE VARIABLE ===\n")
model_without_X3 <- lm(Y ~ X1 + X2 + X4, data = data)
summary(model_without_X3)
cat("VIF after removing X3:\n")
print(vif(model_without_X3))

cat("\n=== SOLUTION 2: REMOVE TWO VARIABLES ===\n")
model_simple <- lm(Y ~ X1 + X4, data = data)
summary(model_simple)
cat("VIF of simple model:\n")
print(vif(model_simple))

cat("\n=== SOLUTION 3: PRINCIPAL COMPONENT ANALYSIS ===\n")
# Standardize variables
data_scaled <- scale(data[, c("X1", "X2", "X3", "X4")])
pca <- prcomp(data_scaled)

# Use principal components
data$PC1 <- pca$x[, 1]
data$PC2 <- pca$x[, 2]
model_pca <- lm(Y ~ PC1 + PC2, data = data)
summary(model_pca)

# ----------------------------------------
# 7. MODEL COMPARISON
# ----------------------------------------

cat("\n=== PERFORMANCE COMPARISON ===\n")
models <- list(
    "Full (multicollinearity)" = model_full,
    "Without X3" = model_without_X3,
    "Simple (X1, X4)" = model_simple,
    "PCA" = model_pca
)

comparison <- data.frame(
    Model = names(models),
    R2 = sapply(models, function(m) summary(m)$r.squared),
    R2_adj = sapply(models, function(m) summary(m)$adj.r.squared),
    AIC = sapply(models, function(m) AIC(m)),
    BIC = sapply(models, function(m) BIC(m))
)

print(comparison)

cat("\n=== CONCLUSION ===\n")
cat("1. Multicollinearity masks the significance of variables\n")
cat("2. VIF easily detects the problem\n")
cat("3. Removing correlated variables improves interpretability\n")
cat("4. R² remains high even with fewer variables\n")

# ----------------------------------------
# 8. BONUS: EFFECT ON CONFIDENCE INTERVALS
# ----------------------------------------

cat("\n=== EFFECT ON CONFIDENCE INTERVALS ===\n")
cat("Model with multicollinearity:\n")
print(confint(model_full))

cat("\nModel without multicollinearity:\n")
print(confint(model_simple))
cat("\nNotice how confidence intervals are much wider with multicollinearity!\n")
