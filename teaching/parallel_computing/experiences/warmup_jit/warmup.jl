#!/usr/bin/env julia
# Experiment "JIT warmup" — Julia.
#
# We time R successive calls of the *same* sum over a CPU vector and observe the
# decay of the time per call. Two variants:
#
#   - mysum  : explicit native loop, compiled just in time → big spike on the 1st
#              call (JIT compilation), then a fast steady state. Analogue of the
#              numba @njit version on the Python side.
#   - sum    : Base's `sum(x)` → already-compiled reference, flat and fast
#              (analogue of numpy.sum).
#
# Zero dependency: Base only, CSV written by hand. Run with: `julia warmup.jl`.
# Common CSV format `iteration,time_s` (see README), read back by plot_warmup.py.

using Random

# --- common parameters (same values as warmup.py) --------------------------
const N = 1_000_000      # vector size (Float64)
const R = 2000           # timed iterations
const SEED = 0

# All results go to warmup_jit/results/ (anchored to the script's folder).
const RESULTS = joinpath(@__DIR__, "results")

function mysum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]
    end
    return s
end

"Returns the vector of times (s) of `reps` successive calls of fn(x).

Julia subtlety: if we called `fn(x)` directly, compiling `measure` would specialize
`fn` and compile the measured function *before* the loop → the JIT spike of the 1st
call would not be timed (unlike numba, which compiles on the 1st call). We therefore
route the call through an **abstract** `Function[fn]` container: the call becomes a
dynamic dispatch, the compilation of the function is deferred to the 1st actual call
and falls inside the `time_ns()`. The dispatch overhead (~ns) is negligible compared to
the steady state (~0.4 ms). This detour also prevents the optimizer from eliminating the
computation."
function measure(fn, x, reps)
    f = Function[fn]   # abstract container → dynamic call, deferred compilation
    times = Vector{Float64}(undef, reps)
    sink = 0.0
    for k in 1:reps
        t0 = time_ns()
        s = f[1](x)
        t1 = time_ns()
        sink += s
        times[k] = (t1 - t0) * 1e-9
    end
    return times, sink
end

function write_csv(path, times)
    open(path, "w") do f
        println(f, "iteration,time_s")
        for (i, t) in enumerate(times)
            # 0-indexed iteration, consistent with the Python version
            println(f, "$(i - 1),", string(t))
        end
    end
end

function summary(name, times)
    first_call = times[1]
    tail = sort(times[end - max(1, length(times) ÷ 10) + 1:end])
    stable = tail[length(tail) ÷ 2 + 1]  # median of the last 10 %
    ratio = stable > 0 ? first_call / stable : Inf
    println("  ", rpad(name, 14),
            " 1st call = ", lpad(round(first_call * 1e3, digits = 3), 9), " ms | ",
            "stable = ", lpad(round(stable * 1e3, digits = 4), 9), " ms | ",
            "ratio = ", lpad(round(ratio, digits = 1), 8), "x")
end

function main()
    mkpath(RESULTS)
    rng = MersenneTwister(SEED)
    x = randn(rng, N)  # allocated once: we measure the warmup, not the allocation

    println("JIT warmup experiment (Julia) — N=$N, R=$R")

    for (tag, fn) in (("julia_mysum", mysum), ("julia_basesum", sum))
        times, sink = measure(fn, x, R)
        write_csv(joinpath(RESULTS, "warmup_$(tag).csv"), times)
        summary(tag, times)
        println("    -> results/warmup_$(tag).csv  (sink=$(sink))")
    end
end

main()
