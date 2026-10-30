# =============================================================================
# Interpreted vs compiled — why the same computation can be 100× faster
# ⚠ KEEP IN SYNC with `pluto/1_compilation_and_types.jl` (same code, two formats).
# Cells: `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools.
# For each cell: predict, then run.
# =============================================================================

#%% 0. Getting started
using BenchmarkTools
using InteractiveUtils   # @code_lowered/_typed/_native/_warntype
@btime 1 + 2             # ~1 ns... yet the cell took seconds. Why?

#%% 1. The JIT warmup — a hand-written loop on 10M floats
N = 10_000_000
v = rand(N)

function mysum(x)
    s = zero(eltype(x))  # a zero of the element type (0.0 here)
    for i in eachindex(x)  # the valid indices of x
        s += x[i]
    end
    return s
end
mysum(v)
# Same loop in pure Python: ~0.5 s. Bet on Julia's time (measured in §2).

#%% The warmup over 10 calls: which call is slow, and why only that one?
let                                 # let … end = a local scope
    function fresh(x)
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]
        end
        return s
    end
    fb = Function[fresh]            # hides the type: compilation lands in call 1
    times = [@elapsed(fb[1](v)) for _ in 1:10]
    println("ms per call: ", round.(times .* 1e3, digits=3))
end
# Full experiment (5 variants, 2000 calls): experiences/warmup_jit/.
# numpy side (fast inside, slow in a Python loop): python/1_compilation_and_types_python.ipynb.

#%% 2. Measuring properly — compare the two timings
let
    g(x) = sum(abs2, x)
    @time g(v)
    @time g(v)
end

#%% Read the distribution: why is the minimum the number to quote?
@btime mysum($v)
@benchmark mysum($v)
# Then try `@btime mysum(v)` without `$`: what changes? (why: with dispatch, untyped globals)

#%% 3. Type specialization — step 1, lowered code
f(x) = 2x + 1
@code_lowered f(3.0)

#%% Step 2 — typed code: compare Float64 and Int
@code_typed f(3.0)

#%% Step 2, with an Int
@code_typed f(3)

#%% Step 3 — assembly: how many instructions? float vs integer ones?
@code_native debuginfo=:none f(3.0)

#%% Step 3, with an Int
@code_native debuginfo=:none f(3)

#%% 4. Bonus (optional) — Base `sum` vs our loop (vector that fits in cache)
vc = rand(100_000)
@btime mysum($vc)

#%% Base `sum`
@btime sum($vc)

#%% Our loop's assembly: `vaddsd` = scalar double. How many accumulators?
@code_native debuginfo=:none mysum(vc)

#%% `sum`'s assembly: look for `vaddpd`... then for the `call`
@code_native debuginfo=:none sum(vc)

#%% The kernel `sum` calls: what replaces `vaddsd`? How many accumulators?
@code_native debuginfo=:none Base.mapreduce_impl(identity, Base.add_sum, vc, 1, length(vc), 1024)
# Why can't the compiler do this to OUR loop? Hint: is (a+b)+c == a+(b+c) in floats?

#%% @simd: permission to reassociate
function mysum_simd(x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)   # @inbounds: skip bounds checks
        s += x[i]
    end
    return s
end
@btime mysum_simd($vc)

#%% 5. Monte-Carlo π — same count as in the Julia-basics module
function count_in_disk(n)
    hits = 0
    for _ in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 ≤ 1
            hits += 1
        end
    end
    return hits
end
estimate_pi(n) = 4 * count_in_disk(n) / n
estimate_pi(10_000_000)

#%% Q1 — how long? how many allocations?
@btime estimate_pi(10_000_000)

#%% Q2 — type-stable? Is the yellow `Union{Nothing, Tuple…}` a problem? Why always Float64?
# @code_warntype = @code_typed with uncertain types in color (red `Any` = unknown).
@code_warntype count_in_disk(1000)
Base.return_types(estimate_pi, (Int,))

#%% Q3 — how does the error shrink with n? Why is this ideal for parallelism?
for n in (10^4, 10^5, 10^6, 10^7, 10^8)
    err = abs(estimate_pi(n) - π)
    println("n = 10^$(round(Int, log10(n)))   error = $(round(err, digits=5))")
end
