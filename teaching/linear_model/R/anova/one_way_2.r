# Linear Model with Sum-to-Zero Constraints
# Two-factor additive model analysis

# Create the data
set.seed(123)
data <- data.frame(
    growth = c(
        20, 22, 19, 21, 25, 27, 24, 26, 18, 20, 17, 19,
        28, 30, 27, 29, 32, 34, 31, 33, 22, 24, 21, 23
    ),
    fertilizer = rep(c("A", "B", "C"), each = 8),
    water = rep(rep(c("Low", "High"), each = 4), 3)
)

# Convert to factors
data$fertilizer <- factor(data$fertilizer)
data$water <- factor(data$water)
mean(data$growth[data$fertilizer == "B"])
# ============================================
# 1. DEFAULT MODEL (Treatment Contrasts)
# ============================================
"========== DEFAULT MODEL (Treatment Contrasts) =========="
model_default <- lm(growth ~ fertilizer + water, data = data)
summary(model_default)

# Computing means with treatment contrasts
"--- Predicted Means (Treatment Contrasts) ---"
# Reference: Fertilizer A, Water High
intercept <- coef(model_default)[1]
fertB_effect <- coef(model_default)[2]
fertC_effect <- coef(model_default)[3]
waterLow_effect <- coef(model_default)[4]

paste("Fertilizer A, Water High:", intercept)
paste("Fertilizer A, Water Low:", intercept + waterLow_effect)
paste("Fertilizer B, Water High:", intercept + fertB_effect)
paste("Fertilizer B, Water Low:", intercept + fertB_effect + waterLow_effect)
paste("Fertilizer C, Water High:", intercept + fertC_effect)
paste("Fertilizer C, Water Low:", intercept + fertC_effect + waterLow_effect)

# ============================================
# 2. SUM-TO-ZERO CONSTRAINTS MODEL
# ============================================
"========== SUM-TO-ZERO CONSTRAINTS MODEL =========="
model_sum <- lm(growth ~ fertilizer + water,
    data = data,
    contrasts = list(
        fertilizer = "contr.sum",
        water = "contr.sum"
    )
)
summary(model_sum)


model_interaction <- lm(growth ~ fertilizer:water, data = data)
summary(model_interaction)
# Extract coefficients
grand_mean <- coef(model_sum)[1]
fertA_effect <- coef(model_sum)[2]
fertB_effect <- coef(model_sum)[3]
fertC_effect <- (fertA_effect + fertB_effect) # Sum to zero constraint
waterLow_effect_sum <- coef(model_sum)[4]
waterHigh_effect_sum <- -waterLow_effect_sum # Sum to zero constraint

"--- Effects Interpretation (Sum-to-Zero) ---"
paste("Grand Mean:", grand_mean)
paste("Fertilizer A effect:", fertA_effect)
paste("Fertilizer B effect:", fertB_effect)
paste("Fertilizer C effect (calculated):", fertC_effect)
paste("Water Low effect:", waterLow_effect_sum)
paste("Water High effect (calculated):", waterHigh_effect_sum)

# ============================================
# 3. MARGINAL MEANS (averaged across water levels)
# ============================================
"========== MARGINAL MEANS FOR FERTILIZERS =========="
"(Averaged across water levels)"

# From the model
"--- From Model with Sum Constraints ---"
mean_A <- grand_mean + fertA_effect
mean_B <- grand_mean + fertB_effect
mean_C <- grand_mean + fertC_effect

paste("Fertilizer A marginal mean:", mean_A)
paste("Fertilizer B marginal mean:", mean_B)
paste("Fertilizer C marginal mean:", mean_C)

# From raw data for verification
"--- From Raw Data (verification) ---"
marginal_means <- aggregate(growth ~ fertilizer, data = data, mean)
marginal_means

# ============================================
# 4. ALL CELL MEANS (Sum-to-Zero Model)
# ============================================
"========== ALL CELL MEANS (Sum-to-Zero Model) =========="
"Mean = Grand Mean + Fertilizer Effect + Water Effect"

# Create a matrix of predicted means
fert_levels <- c("A", "B", "C")
water_levels <- c("Low", "High")
fert_effects <- c(fertA_effect, fertB_effect, fertC_effect)
water_effects <- c(waterLow_effect_sum, waterHigh_effect_sum)

predicted_means <- matrix(NA, nrow = 3, ncol = 2)
rownames(predicted_means) <- fert_levels
colnames(predicted_means) <- water_levels

for (i in 1:3) {
    for (j in 1:2) {
        predicted_means[i, j] <- grand_mean + fert_effects[i] + water_effects[j]
    }
}

predicted_means

# Verify with actual group means
"--- Actual Group Means from Data ---"
actual_means <- aggregate(growth ~ fertilizer + water, data = data, mean)
actual_means_matrix <- reshape(actual_means,
    idvar = "fertilizer",
    timevar = "water",
    direction = "wide"
)
names(actual_means_matrix) <- c("fertilizer", "Low", "High")
actual_means_matrix

# ============================================
# 5. INTERACTION MODEL WITH SUM CONSTRAINTS
# ============================================
"========== INTERACTION MODEL (Sum-to-Zero Constraints) =========="
model_interaction <- lm(growth ~ fertilizer * water,
    data = data,
    contrasts = list(
        fertilizer = "contr.sum",
        water = "contr.sum"
    )
)
summary(model_interaction)

# Extract coefficients
grand_mean_int <- coef(model_interaction)[1]
fertA_main <- coef(model_interaction)[2]
fertB_main <- coef(model_interaction)[3]
fertC_main <- -(fertA_main + fertB_main)
waterLow_main <- coef(model_interaction)[4]
waterHigh_main <- -waterLow_main

# Interaction effects (only 2 are shown, others are constrained)
int_A_Low <- coef(model_interaction)[5]
int_B_Low <- coef(model_interaction)[6]

# Calculate remaining interaction effects using constraints
# Row constraints: interactions sum to 0 across fertilizers for each water level
int_C_Low <- -(int_A_Low + int_B_Low)

# Column constraints: interactions sum to 0 across water levels for each fertilizer
int_A_High <- -int_A_Low
int_B_High <- -int_B_Low
int_C_High <- -int_C_Low

"--- Main Effects and Interactions (Sum-to-Zero) ---"
paste("Grand Mean:", grand_mean_int)

"Main Effects:"
paste("Fertilizer A:", fertA_main)
paste("Fertilizer B:", fertB_main)
paste("Fertilizer C (calculated):", fertC_main)
paste("Water Low:", waterLow_main)
paste("Water High (calculated):", waterHigh_main)

"Interaction Effects:"
paste("A:Low:", int_A_Low, "  A:High:", int_A_High)
paste("B:Low:", int_B_Low, "  B:High:", int_B_High)
paste("C:Low:", int_C_Low, "  C:High:", int_C_High)

# ============================================
# 6. CELL MEANS WITH INTERACTION
# ============================================
"========== CELL MEANS (Interaction Model) =========="
"Mean = Grand Mean + Fert Main + Water Main + Interaction"

# Create matrix of predicted means with interaction
fert_mains <- c(fertA_main, fertB_main, fertC_main)
water_mains <- c(waterLow_main, waterHigh_main)
interactions <- matrix(
    c(
        int_A_Low, int_A_High,
        int_B_Low, int_B_High,
        int_C_Low, int_C_High
    ),
    nrow = 3, byrow = TRUE
)

predicted_means_int <- matrix(NA, nrow = 3, ncol = 2)
rownames(predicted_means_int) <- fert_levels
colnames(predicted_means_int) <- water_levels

for (i in 1:3) {
    for (j in 1:2) {
        predicted_means_int[i, j] <- grand_mean_int + fert_mains[i] +
            water_mains[j] + interactions[i, j]
    }
}

"--- Predicted Means from Interaction Model ---"
predicted_means_int

"--- Actual Group Means (for comparison) ---"
actual_means_matrix

# ============================================
# 7. MARGINAL MEANS WITH INTERACTION
# ============================================
"========== MARGINAL MEANS (Interaction Model) =========="

# For marginal means, interactions cancel out due to sum-to-zero constraints
"--- Marginal Means for Fertilizers ---"
"(Note: Interactions sum to zero across water levels)"
marginal_A_int <- grand_mean_int + fertA_main
marginal_B_int <- grand_mean_int + fertB_main
marginal_C_int <- grand_mean_int + fertC_main

paste("Fertilizer A:", marginal_A_int)
paste("Fertilizer B:", marginal_B_int)
paste("Fertilizer C:", marginal_C_int)

"--- Marginal Means for Water ---"
"(Note: Interactions sum to zero across fertilizers)"
marginal_Low <- grand_mean_int + waterLow_main
marginal_High <- grand_mean_int + waterHigh_main

paste("Water Low:", marginal_Low)
paste("Water High:", marginal_High)

# ============================================
# 8. COMPARISON: ADDITIVE vs INTERACTION
# ============================================
"========== MODEL COMPARISON =========="
"Testing if interaction is significant:"
anova(model_sum, model_interaction)
