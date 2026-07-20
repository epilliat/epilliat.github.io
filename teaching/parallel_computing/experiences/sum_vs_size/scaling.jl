#!/usr/bin/env julia
# Experiment "sum vs size" — Julia.
#
# Time of a sum as a function of the number of elements N of the vector, in the steady
# state (compilation done once and for all before the loop — see the warmup_jit experiment).
# Two variants:
#
#   - mysum : explicit native loop (scalar) → ~ linear and smooth time in N;
#   - sum   : Base's `sum(x)` (pairwise summation + SIMD) → reveals a slight
#             **staircase** (the steps of the recursion / of the SIMD blocking).
#
# For each N we take REPS measurements → mean ± standard deviation (the CI drawn on the slides).
# Zero dependency (Base only; mean/standard deviation computed by hand). Run with: `julia scaling.jl`.
# CSV output `n,mean_s,std_s` in results/, read back by plot_scaling.py.

using Random

# --- parameters ------------------------------------------------------------
const N_START = 10_000
const N_STOP  = 100_000      # 10^5
const N_STEP  = 100
const REPS    = 10           # measurements per point
const SEED    = 0

const RESULTS = joinpath(@__DIR__, "results")

function mysum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]
    end
    return s
end

"REPS times (s) of successive calls of fn(x), in the steady state (fn already compiled)."
function sample_times(fn, x, reps)
    ts = Vector{Float64}(undef, reps)
    sink = 0.0
    for k in 1:reps
        t0 = time_ns()
        s = fn(x)
        t1 = time_ns()
        sink += s
        ts[k] = (t1 - t0) * 1e-9
    end
    return ts, sink
end

mean(v) = sum(v) / length(v)

function std(v)                      # sample standard deviation (n-1)
    m = mean(v)
    return sqrt(sum((x - m)^2 for x in v) / (length(v) - 1))
end

function write_csv(path, ns, means, stds)
    open(path, "w") do f
        println(f, "n,mean_s,std_s")
        for i in eachindex(ns)
            println(f, ns[i], ",", means[i], ",", stds[i])
        end
    end
end

function main()
    mkpath(RESULTS)
    rng = MersenneTwister(SEED)

    # Compilation out of the measurement: one call of each function on a small vector.
    let xw = randn(rng, 1000)
        sample_times(mysum, xw, 3)
        sample_times(sum, xw, 3)
    end

    ns = collect(N_START:N_STEP:N_STOP)
    sink = 0.0
    results = Dict("julia_mysum" => (Float64[], Float64[]),
                   "julia_basesum" => (Float64[], Float64[]))

    println("Sum vs size experiment (Julia) — N ∈ $N_START:$N_STEP:$N_STOP ($(length(ns)) points), REPS=$REPS")

    for n in ns
        x = randn(rng, n)                       # same vector for both variants
        for (tag, fn) in (("julia_mysum", mysum), ("julia_basesum", sum))
            ts, s = sample_times(fn, x, REPS)
            sink += s
            push!(results[tag][1], mean(ts))
            push!(results[tag][2], std(ts))
        end
    end

    for tag in ("julia_mysum", "julia_basesum")
        means, stds = results[tag]
        write_csv(joinpath(RESULTS, "scaling_$(tag).csv"), ns, means, stds)
        mid = length(ns) ÷ 2
        println("  ", rpad(tag, 14),
                " @N=$(ns[mid]) : ", round(means[mid] * 1e6, digits = 2), " µs ",
                "± ", round(stds[mid] * 1e6, digits = 2), " µs")
        println("    -> results/scaling_$(tag).csv")
    end
    println("  (sink=$(sink))")
end

main()
