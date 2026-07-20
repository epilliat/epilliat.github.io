#!/usr/bin/env julia
# Code reflection — why Base's `sum` beats `mysum`.
#
# Dumps the native code (Intel asm) of `mysum`, of Base's `sum` and of its
# `mapreduce_impl` kernel, then times `mysum` / `mysum_simd` / `sum` in the steady state.
# This is the source of the asm excerpts and of the numbers shown in chapter 1
# (slides/1_compilation_and_types.qmd). Zero dependency (InteractiveUtils is stdlib).
#
# Run with: `julia reflection.jl`  → writes results/*.asm and prints the benchmark.

using InteractiveUtils

const RESULTS = joinpath(@__DIR__, "results")

# "Naive" scalar loop: ordered addition, a single accumulator.
function mysum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]
    end
    return s
end

# Same loop, but @simd allows reassociation → vectorization + multiple accumulators.
function mysum_simd(x)
    s = 0.0
    @inbounds @simd for i in eachindex(x)
        s += x[i]
    end
    return s
end

dump_native(io, f, types) =
    code_native(io, f, types; syntax = :intel, debuginfo = :none)

function write_asm()
    mkpath(RESULTS)
    open(io -> dump_native(io, mysum, (Vector{Float64},)),
         joinpath(RESULTS, "mysum.asm"), "w")
    open(io -> dump_native(io, sum, (Vector{Float64},)),
         joinpath(RESULTS, "sum.asm"), "w")
    # The real vectorized kernel called by `sum` (N ≥ 16): pairwise + SIMD.
    open(io -> dump_native(io, Base.mapreduce_impl,
                           (typeof(identity), typeof(Base.add_sum),
                            Vector{Float64}, Int, Int, Int)),
         joinpath(RESULTS, "mapreduce_impl.asm"), "w")
    println("-> results/{mysum,sum,mapreduce_impl}.asm")
end

# µs/call in the steady state (warmup already done, result accumulated to avoid DCE).
function bench(f, x, n)
    s = 0.0
    t0 = time_ns()
    for _ in 1:n
        s += f(x)
    end
    return (time_ns() - t0) / n * 1e-3, s
end

function main()
    write_asm()

    x = randn(100_000)
    mysum(x); mysum_simd(x); sum(x)        # warmup (compilation out of the measurement)
    n = 2000
    println("\nBenchmark (N=$(length(x)), $n calls):")
    for (name, f) in (("mysum", mysum), ("mysum_simd", mysum_simd), ("sum (Base)", sum))
        t, _ = bench(f, x, n)
        println("  ", rpad(name, 11), " : ", round(t, digits = 2), " µs")
    end

    # Quick tally of the key instructions (scalar vs packed SIMD).
    function tally(path)
        txt = read(path, String)
        n_sd = count("vaddsd", txt)
        n_pd = count("vaddpd", txt)
        ymm = !isempty(findall("ymm", txt))
        println("  ", rpad(basename(path), 20),
                " vaddsd=", n_sd, "  vaddpd=", n_pd, "  ymm=", ymm)
    end
    println("\nInstructions (scalar = vaddsd, packed SIMD = vaddpd):")
    tally(joinpath(RESULTS, "mysum.asm"))
    tally(joinpath(RESULTS, "mapreduce_impl.asm"))
end

main()
