# Figures for the "trees" example used in slides/inference.qmd
#   images/trees1.png                      -- scatter plot Volume ~ Girth
#   images/trees_regression_intervals.png  -- fit + confidence and prediction bands
#
# The slides are shown light-on-white (dark mode presents images as white cards),
# so the figures keep a white background and reuse the palette of the in-slide
# animations: dark slate for the data, red for the fit, blue/green for the bands.

library(datasets)
require(stats)
require(graphics)

col_pt <- "#2c3e50"
col_fit <- "#e74c3c"
col_conf <- "#3b6fb6"
col_pred <- "#1e8449"
col_grid <- "#e3e7ee"
col_axis <- "#5a6678"
col_lab <- "#1f2733"

open_png <- function(file, width = 8, height = 5) {
     png(file,
          width = width, height = height, units = "in",
          res = 200, type = "cairo", bg = "white"
     )
     par(
          mar = c(4.4, 4.6, 1.4, 1.2), mgp = c(2.8, 0.7, 0), las = 1,
          cex = 1.15, cex.lab = 1.05, family = "sans", bty = "n", tcl = -0.3,
          col.axis = col_axis, col.lab = col_lab, fg = col_axis
     )
}

# light grid drawn under the data, then thin axes
frame_plot <- function(xat, yat) {
     abline(v = xat, h = yat, col = col_grid, lwd = 1)
     axis(1, at = xat, lwd = 0, lwd.ticks = 1)
     axis(2, at = yat, lwd = 0, lwd.ticks = 1)
}

points_data <- function(x, y, cex = 1.3) {
     points(x, y,
          pch = 21, cex = cex, lwd = 1.1,
          bg = adjustcolor(col_pt, 0.72), col = "white"
     )
}

# ---------------------------------------------------------------- scatter plot

xat <- seq(8, 21, by = 2)
yat <- seq(10, 80, by = 10)

open_png("images/trees1.png")
plot(Volume ~ Girth,
     data = trees, type = "n", axes = FALSE,
     xlim = c(8, 21), ylim = c(8, 80),
     xlab = "Girth (inches)", ylab = "Volume (cubic feet)"
)
frame_plot(xat, yat)
points_data(trees$Girth, trees$Volume)
mtext(sprintf("n = %d trees", nrow(trees)),
     side = 3, adj = 1, line = -0.2, cex = 1.0, col = col_axis
)
dev.off()

# --------------------------------------------- fit with confidence / prediction

reg <- lm(Volume ~ Girth, data = trees)
summary(reg)

x_seq <- seq(8, 21, length.out = 200)
conf_int <- predict(reg,
     newdata = data.frame(Girth = x_seq),
     interval = "confidence", level = 0.95
)
pred_int <- predict(reg,
     newdata = data.frame(Girth = x_seq),
     interval = "prediction", level = 0.95
)

x_0 <- 15
p_0 <- predict(reg, newdata = data.frame(Girth = x_0), interval = "prediction")
c_0 <- predict(reg, newdata = data.frame(Girth = x_0), interval = "confidence")
y_hat <- p_0[, "fit"]

ylim <- range(pred_int, trees$Volume)
yat <- seq(0, 90, by = 15)

open_png("images/trees_regression_intervals.png", width = 9, height = 5.6)
plot(Volume ~ Girth,
     data = trees, type = "n", axes = FALSE,
     xlim = c(8, 21), ylim = ylim,
     xlab = "Girth (inches)", ylab = "Volume (cubic feet)"
)
frame_plot(xat, yat)

# bands, widest first
polygon(c(x_seq, rev(x_seq)), c(pred_int[, "lwr"], rev(pred_int[, "upr"])),
     col = adjustcolor(col_pred, 0.10), border = NA
)
polygon(c(x_seq, rev(x_seq)), c(conf_int[, "lwr"], rev(conf_int[, "upr"])),
     col = adjustcolor(col_conf, 0.20), border = NA
)
lines(x_seq, pred_int[, "lwr"], col = col_pred, lty = 3, lwd = 2)
lines(x_seq, pred_int[, "upr"], col = col_pred, lty = 3, lwd = 2)
lines(x_seq, conf_int[, "lwr"], col = col_conf, lty = 2, lwd = 2)
lines(x_seq, conf_int[, "upr"], col = col_conf, lty = 2, lwd = 2)

abline(reg, col = col_fit, lwd = 2.6)
points_data(trees$Girth, trees$Volume, cex = 1.2)

# the two intervals at one x_0, drawn side by side so both widths are readable
bracket <- function(x, lo, up, col, cap = 0.22) {
     segments(x, lo, x, up, col = col, lwd = 3)
     segments(x - cap, c(lo, up), x + cap, c(lo, up), col = col, lwd = 3)
}
segments(x_0, ylim[1], x_0, y_hat, col = adjustcolor(col_fit, 0.55), lty = 3, lwd = 1.4)
bracket(x_0 - 0.42, p_0[, "lwr"], p_0[, "upr"], col_pred)
bracket(x_0 + 0.42, c_0[, "lwr"], c_0[, "upr"], col_conf)
points(x_0, y_hat, pch = 21, cex = 1.7, lwd = 1.4, bg = col_fit, col = "white")
text(x_0 - 0.8, y_hat + 1.2, expression(hat(Y)[0]), col = col_fit, cex = 1.2, adj = 1)
mtext(expression(x[0]), side = 1, at = x_0, line = 0.9, col = col_fit, cex = 1.1)

legend("topleft",
     legend = c(
          "Data", "Regression line",
          "95% confidence interval (mean)",
          "95% prediction interval (new tree)"
     ),
     col = c(col_pt, col_fit, col_conf, col_pred),
     pt.bg = c(adjustcolor(col_pt, 0.72), NA, NA, NA),
     lty = c(NA, 1, 2, 3), pch = c(21, NA, NA, NA),
     lwd = c(1.1, 2.6, 2, 2), pt.cex = 1.2,
     bty = "n", cex = 0.92, text.col = col_lab, inset = c(0, 0.01),
     y.intersp = 1.25, seg.len = 2.4
)

cat("For Girth =", x_0, "inches, predicted Volume =", round(y_hat, 2), "cubic feet\n")
dev.off()
