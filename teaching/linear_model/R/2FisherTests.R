# ================================================
# POLYNOMIAL REGRESSION AND OVERFITTING DEMONSTRATION
# ================================================


set.seed(42) # for reproducibility

# ----------------------------------------
# 1. GENERATE DATA
# ----------------------------------------

# %%

n <- 50 # number of observations

# True relationship: Y = 5 + 3*X - 2X^2 + noise
# (simple linear relationship with some noise)
X <- runif(n, min = 0, max = 10)
Y <- 5 + 3 * X - 2 * X^2 + rnorm(n, mean = 0, sd = 5)

# Create dataframe
data <- data.frame(X = X, Y = Y)


# %%
model <- lm(Y ~ X + I(X^2) + I(X^3), data = data)

summary(model)

# %%


# Under the black box for Global Fisher Test
R = 
summary(model)$fstatistic[1]
library(pracma)  # for nullspace function
V <- nullspace(R)  # Returns a matrix whose columns form a basis for Ker(R)
# where k = dim(Ker(R)) = p - rank(R)