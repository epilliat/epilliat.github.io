if (!require("datasets")) {
     install.packages("datasets")
     library("datasets")
}

require(stats)
require(graphics)
pairs(trees, main = "trees data")
trees[, c("Girth", "Volume")]


png("images/trees1.png", width = 800, height = 600)
plot(Volume ~ Girth, data = trees, log = "xy")
dev.off()


reg <- lm(Volume ~ Girth, data = trees)
plot(Volume ~ Girth, data = trees, log = "xy")
abline(reg)
summary(reg)
## i.e., Volume ~= c * Height * Girth^2  seems reasonable




# Save the plot
png("images/trees_regression_intervals.png", width = 1000, height = 700)

reg <- lm(Volume ~ Girth, data = trees)

# Create sequence of x values for smooth curves
x_seq <- seq(min(trees$Girth), max(trees$Girth), length.out = 100)

# Calculate confidence and prediction intervals
conf_int <- predict(reg,
     newdata = data.frame(Girth = x_seq),
     interval = "confidence", level = 0.95
)
pred_int <- predict(reg,
     newdata = data.frame(Girth = x_seq),
     interval = "prediction", level = 0.95
)

# Create the plot
plot(Volume ~ Girth,
     data = trees,
     main = "Tree Volume vs Girth with Confidence and Prediction Intervals",
     xlab = "Girth (inches)", ylab = "Volume (cubic feet)",
     pch = 16, col = "black"
)

# Add regression line
abline(reg, col = "red", lwd = 2)

# Add confidence interval
lines(x_seq, conf_int[, "lwr"], col = "blue", lty = 2, lwd = 2)
lines(x_seq, conf_int[, "upr"], col = "blue", lty = 2, lwd = 2)

# Add prediction interval
lines(x_seq, pred_int[, "lwr"], col = "green", lty = 3, lwd = 2)
lines(x_seq, pred_int[, "upr"], col = "green", lty = 3, lwd = 2)

# Choose a specific x_0 value and annotate Y_hat
x_0 <- 15 # Choose girth = 15 inches
y_hat <- predict(reg, newdata = data.frame(Girth = x_0))

# Add point and annotation for Y_hat
points(x_0, y_hat, pch = 16, col = "red", cex = 1.5)
text(x_0 + 0.5, y_hat + 2,
     expression(hat(Y)[o]),
     col = "red", cex = 1.2, font = 2
)

# Add vertical line to show x_0
segments(x_0, 0, x_0, y_hat, col = "red", lty = 4, lwd = 1)

# Add legend
legend("topleft",
     legend = c(
          "Data", "Regression Line", "95% Confidence Interval",
          "95% Prediction Interval", expression(hat(Y)[o])
     ),
     col = c("black", "red", "blue", "green", "red"),
     lty = c(NA, 1, 2, 3, NA),
     pch = c(16, NA, NA, NA, 16),
     lwd = c(NA, 2, 2, 2, NA),
     cex = 0.8
)

# Print the predicted value
cat("For Girth =", x_0, "inches, predicted Volume =", round(y_hat, 2), "cubic feet\n")


# Repeat the entire plotting code here if you want to save
dev.off()




length(trees[, 1])
