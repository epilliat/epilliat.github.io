using Plots, Statistics, Distributions, Random, JuMP, SpecialFunctions

working_chips = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
frequencies = [8, 10, 16, 17, 21, 20, 17, 14, 10, 7, 3, 5, 2] ./ 150


bar(working_chips, frequencies, alpha=0.5, label="empirical")

bar!(working_chips, pdf.(NegativeBinomial(5, 0.5), working_chips), color="red", alpha=0.3, label="NB")


pdf.(NegativeBinomial(4, 0.9), working_chips)

