using HypothesisTests
using Distributions

t = FisherExactTest(4, 0, 0, 4)

pvalue(t)

1 / 70 * 2

using HypothesisTests


# Observed values
O = [60, 40, 30, 50, 55, 25, 30, 45, 55]

# Expected values
E1 = 46.7
E2 = 36.7

# Calculation
chi_square_statistic = ((O[1] - E1)^2 / E1) +
                       ((O[2] - E1)^2 / E1) +
                       ((O[3] - E2)^2 / E2) +
                       ((O[4] - E1)^2 / E1) +
                       ((O[5] - E1)^2 / E1) +
                       ((O[6] - E2)^2 / E2) +
                       ((O[7] - E1)^2 / E1) +
                       ((O[8] - E1)^2 / E1) +
                       ((O[9] - E2)^2 / E2)


using Distributions
1 - cdf(Chisq(4), 26.57)
using Random



function wilcox_table(n)
    u = n * (n + 1) / 2
    d = zero.(1:u)
    N = 100000
    for _ in (1:N)
        k = rand(Binomial(n, 0.5))
        w = sum(randperm(n)[1:k])
        #if w > u / 2
        #    w = u - w
        #end
        d[Int(w)] += 1
    end
    for k in (1:Int(u))
        d[k] /= N
    end
    return d
end


using Plots
n=20
d=wilcox_table(20)
plot(d)
plot!(n*(n+1)/4 + sqrt(n*(n+1)*(2n+1)/24)*Normal(0,1))
using StatsPlots
plot(Normal(0,))
