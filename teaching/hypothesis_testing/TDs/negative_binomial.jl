using Plots, Statistics, Distributions, Random, JuMP, SpecialFunctions

working_chips = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
frequencies = [8, 10, 16, 17, 21, 20, 17, 14, 10, 7, 3, 5, 2] ./ 150


bar(working_chips, frequencies, alpha=0.5, label="empirical")
bar!(working_chips, pdf.(NegativeBinomial(5, 0.2), working_chips), color="red", alpha=0.3, label="NB")



p = 0.9 # Theoretical proba of working chips
r = 5 # Number of failing chips per batch

plot(working_chips, pdf.(NegativeBinomial(r, 1 - p), working_chips))
plot!(working_chips, [binomial(5 + x - 1, x) * p^x * (1 - p)^r for x in working_chips], label="probas of BN(r, 1-p)")

number_of_working_chips = sum(working_chips .* frequencies * 150)
number_of_failing_chips = 150 * r


hatp = number_of_working_chips / (number_of_working_chips + number_of_failing_chips)


bar(working_chips, frequencies, alpha=0.5, label="empirical")
bar!(working_chips, pdf.(NegativeBinomial(5, 1 - hatp), working_chips), color="red", alpha=0.3, label="NB")



O = frequencies * 150
E = pdf.(NegativeBinomial(r, 1 - hatp), working_chips) * 150


sum((O - E) .^ 2 ./ E)
