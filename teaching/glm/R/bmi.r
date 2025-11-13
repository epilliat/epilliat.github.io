# ============================================================================
# Simulated BMI Study Data and Logistic Regression Analysis
# ============================================================================

set.seed(42)  # For reproducibility

# Generate data for 5300 patients
n <- 5300

# Generate predictor variables
AGE <- rnorm(n, mean = 50, sd = 12)
AGE <- pmax(pmin(AGE, 80), 20)  # Constrain between 20 and 80

DBP <- rnorm(n, mean = 80, sd = 10)  # Diastolic blood pressure
DBP <- pmax(pmin(DBP, 110), 60)  # Constrain between 60 and 110

SEXE <- factor(sample(c("HOMME", "FEMME"), n, replace = TRUE, prob = c(0.48, 0.52)))

ACTIV <- rbinom(n, 1, prob = 0.25)  # 25% do intense sports

WALK <- rbinom(n, 1, prob = 0.35)   # 35% walk/cycle to work

MARITAL <- factor(sample(1:6, n, replace = TRUE, 
                        prob = c(0.45, 0.05, 0.15, 0.03, 0.20, 0.12)))
# 1=married, 2=widowed, 3=divorced, 4=separated, 5=single, 6=cohabiting

# Generate outcome Y (BMI > 35) based on predictors
# Using a logistic model with non-linear age effect and interaction-like patterns

# Create linear predictor with AGE^2 effect
linear_pred <- -3.95 + 
               0.064 * AGE - 
               0.00068 * AGE^2 + 
               0.012 * DBP + 
               0.52 * (SEXE == "FEMME") - 
               0.40 * WALK - 
               0.66 * ACTIV +
               rnorm(n, 0, 0.5)  # Add some noise

# Convert to probability
prob_Y <- 1 / (1 + exp(-linear_pred))

# Generate binary outcome
Y <- rbinom(n, 1, prob = prob_Y)

# Create data frame
bmi_data <- data.frame(
  Y = Y,
  AGE = AGE,
  DBP = DBP,
  SEXE = SEXE,
  ACTIV = ACTIV,
  WALK = WALK,
  MARITAL = MARITAL
)


summary(bmi_data)

# ============================================================================
# MODEL 1: With all variables including MARITAL
# ============================================================================

model1 <- glm(Y ~ AGE + DBP + SEXE + ACTIV + WALK + MARITAL, 
              family=binomial,
              
              
              
              data = bmi_data)

summary(model1)


coef_table1 <- summary(model1)$coefficients
print(round(coef_table1, 6))

# ============================================================================
# MODEL 2: With AGE^2, without MARITAL
# ============================================================================

cat("\n\n=== MODEL 2: Model with AGE^2, without MARITAL ===\n")
model2 <- glm(Y ~ AGE + I(AGE^2) + DBP + SEXE + WALK + ACTIV, 
              family = binomial, 
              data = bmi_data)

summary(model2)

# Display coefficients table
cat("\n--- Coefficients Table ---\n")
coef_table2 <- summary(model2)$coefficients
print(round(coef_table2, 7))

# ============================================================================
# PREDICTIONS: Examples from the slides
# ============================================================================

cat("\n\n=== PREDICTION EXAMPLES ===\n")

# Example 1: WALK=0, ACTIV=0, AGE=55, DBP=85
new_data1 <- data.frame(
  AGE = 55,
  DBP = 85,
  SEXE = factor("HOMME", levels = c("HOMME", "FEMME")),
  WALK = 0,
  ACTIV = 0
)

pred1 <- predict(model2, newdata = new_data1, type = "response")
cat("\nPerson 1: AGE=55, DBP=85, WALK=0, ACTIV=0 (male, no exercise)\n")
cat("Predicted P(Y=1):", round(pred1, 4), "\n")

# Manual calculation to verify
linear_pred1 <- coef(model2)[1] + 
                coef(model2)[2] * 55 + 
                coef(model2)[3] * 55^2 + 
                coef(model2)[4] * 85
prob1 <- 1 / (1 + exp(-linear_pred1))
cat("Manual calculation: ", round(prob1, 4), "\n")

# Example 2: WALK=0, ACTIV=1, AGE=55, DBP=85
new_data2 <- data.frame(
  AGE = 55,
  DBP = 85,
  SEXE = factor("HOMME", levels = c("HOMME", "FEMME")),
  WALK = 0,
  ACTIV = 1
)

pred2 <- predict(model2, newdata = new_data2, type = "response")
cat("\nPerson 2: AGE=55, DBP=85, WALK=0, ACTIV=1 (male, intense sports)\n")
cat("Predicted P(Y=1):", round(pred2, 4), "\n")

# Manual calculation to verify
linear_pred2 <- linear_pred1 + coef(model2)[7]
prob2 <- 1 / (1 + exp(-linear_pred2))
cat("Manual calculation: ", round(prob2, 4), "\n")
cat("Difference due to ACTIV:", round(pred1 - pred2, 4), "\n")

# ============================================================================
# VISUALIZATION: Probability curves
# ============================================================================

cat("\n\n=== GENERATING VISUALIZATIONS ===\n")

# Create a grid of AGE values for plotting
age_grid <- seq(20, 80, length.out = 100)

# Predictions for different scenarios (DBP=80, male)
pred_data <- data.frame(
  AGE = rep(age_grid, 4),
  DBP = 80,
  SEXE = factor("HOMME", levels = c("HOMME", "FEMME")),
  WALK = rep(c(0, 0, 1, 1), each = 100),
  ACTIV = rep(c(0, 1, 0, 1), each = 100)
)

pred_data$prob <- predict(model2, newdata = pred_data, type = "response")
pred_data$group <- paste0("WALK=", pred_data$WALK, ", ACTIV=", pred_data$ACTIV)

# Plot
library(ggplot2)

p <- ggplot(pred_data, aes(x = AGE, y = prob, color = group, linetype = group)) +
  geom_line(linewidth = 1.2) +
  labs(
    title = "Predicted Probability of BMI > 35 by Age",
    subtitle = "For males with DBP = 80",
    x = "Age (years)",
    y = "P(BMI > 35)",
    color = "Activity Profile",
    linetype = "Activity Profile"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2))

print(p)

# Save plot
ggsave("bmi_predictions.png", p, width = 10, height = 6, dpi = 300)
cat("Plot saved as 'bmi_predictions.png'\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

cat("\n\n=== MODEL COMPARISON ===\n")
cat("AIC Model 1 (with MARITAL):", round(AIC(model1), 2), "\n")
cat("AIC Model 2 (with AGE^2, no MARITAL):", round(AIC(model2), 2), "\n")
cat("\nLower AIC indicates better model fit.\n")

# Likelihood ratio test to compare models
# Note: Models must be nested for this to be valid
cat("\n--- Significance of MARITAL variable ---\n")
model1_no_marital <- glm(Y ~ AGE + DBP + SEXE + ACTIV + WALK, 
                         family = binomial, 
                         data = bmi_data)
lrt <- anova(model1_no_marital, model1, test = "Chisq")
print(lrt)

# ============================================================================
# INTERPRETATION OF COEFFICIENTS
# ============================================================================

cat("\n\n=== INTERPRETATION OF KEY COEFFICIENTS (Model 2) ===\n")

cat("\nDBP (Diastolic Blood Pressure):\n")
cat("  Coefficient:", round(coef(model2)["DBP"], 4), "\n")
cat("  Odds ratio:", round(exp(coef(model2)["DBP"]), 4), "\n")
cat("  Interpretation: Each 1 mmHg increase in DBP multiplies odds of BMI>35 by",
    round(exp(coef(model2)["DBP"]), 3), "\n")

cat("\nSEXEFEMME (Being Female):\n")
cat("  Coefficient:", round(coef(model2)["SEXEFEMME"], 4), "\n")
cat("  Odds ratio:", round(exp(coef(model2)["SEXEFEMME"]), 4), "\n")
cat("  Interpretation: Females have", round(exp(coef(model2)["SEXEFEMME"]), 2), 
    "times the odds of BMI>35 compared to males\n")

cat("\nWALK1 (Walking/Cycling to work):\n")
cat("  Coefficient:", round(coef(model2)["WALK"], 4), "\n")
cat("  Odds ratio:", round(exp(coef(model2)["WALK"]), 4), "\n")
cat("  Interpretation: Walking/cycling reduces odds of BMI>35 by",
    round((1 - exp(coef(model2)["WALK"])) * 100, 1), "%\n")

cat("\nACTIV1 (Intense sports activity):\n")
cat("  Coefficient:", round(coef(model2)["ACTIV"], 4), "\n")
cat("  Odds ratio:", round(exp(coef(model2)["ACTIV"]), 4), "\n")
cat("  Interpretation: Intense sports reduces odds of BMI>35 by",
    round((1 - exp(coef(model2)["ACTIV"])) * 100, 1), "%\n")

cat("\n=== ANALYSIS COMPLETE ===\n")