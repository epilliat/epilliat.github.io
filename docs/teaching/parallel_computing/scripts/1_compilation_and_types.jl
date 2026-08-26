# =============================================================================
# Interpreted vs compiled — why the same computation can be 100× faster
# ENSAI 3A — Julia as a test bench, Python as a point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC: this is the plain-script twin of `pluto/1_compilation_and_types.jl`.
#   The Pluto notebook and this script hold the SAME lesson in two formats — edit
#   BOTH together whenever the content changes, so they never drift apart.
# -----------------------------------------------------------------------------
# Cells are delimited by `#%%`. In VS Code (Julia extension) run a cell with
# Alt+Enter / Ctrl+Enter — the value of the last expression shows inline and the
# variables stay in the REPL. Needs: BenchmarkTools, Printf (install once with
#   using Pkg; Pkg.add(["BenchmarkTools"])   # Printf & Random are stdlib
#
# The thread of the whole course: WHY can the same computation, mathematically
# identical, be tens or hundreds of times faster depending on how it's written?
# Three ingredients today: interpreted vs compiled · type specialization ·
# measuring properly. Parallelism (threads, GPU) comes later — but you can't
# parallelize code whose sequential cost you don't yet understand.
# =============================================================================

#%% 0. Getting started — load the tools
using BenchmarkTools     # @btime, @benchmark: reliable measurements
using InteractiveUtils   # @code_lowered/_typed/_native/_warntype (auto-loaded in the REPL)
using Printf             # @sprintf for aligned output
using Profile            # @profile: the sampling profiler (section 7)
using Random             # MersenneTwister (reproducible darts)
# The FIRST use of a package may take a moment (precompilation): already an
# illustration of the course — Julia compiles.
@btime 1 + 2

#%% Same, with the full distribution
@benchmark 1 + 2

#%% 1. Same computation, very different times
# Simplest possible demo: summing a large vector of Float64. Written three ways —
# a pure Python loop (compared mentally), numpy, and a Julia loop. The computation
# is EXACTLY the same. The times are not.
const N = 10_000_000
const v = rand(N)
summary(v)

#%% A hand-written Julia loop
function mysum(x)
    s = zero(eltype(x))
    for i in eachindex(x)
        s += x[i]
    end
    return s
end
mysum(v)   # check it works

#%% The Python point of comparison
# In pure Python the same reads:
#     def mysum(x):
#         s = 0.0
#         for xi in x:
#             s += xi
#         return s
# On 10M elements that Python loop takes ~1 s; the Julia loop above takes a few
# MILLISECONDS — ~100×, for code that looks the same line for line.
# numpy.sum(x) is fast (~10 ms) but because it DOESN'T run in Python: it calls a
# compiled C routine (see section 5). So: why is the Julia loop fast while the
# Python loop is slow?

#%% 2. The warmup: the first call is special
# The very first call pays the JIT compilation; the following calls are the real
# cost. The 1st @time is inflated (allocations, "compilation" time), the 2nd real.
let
    g(x) = sum(abs2, x)        # new function, never compiled
    @time g(v)                 # 1st call: includes compilation
    @time g(v)                 # 2nd call: the real cost
end

#%% The warmup, seen over several calls
# Time EACH call of a fresh function: the 1st pays compilation (a spike), then the
# time decays to the steady state. Trick: call it via an abstract Function[...]
# container so compilation falls INSIDE the 1st timed call.
let
    function fresh(x)
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]
        end
        return s
    end
    fb = Function[fresh]
    times = [@elapsed(fb[1](v)) for _ in 1:10]
    for k in eachindex(times)
        println(@sprintf("call %2d : %8.3f ms", k, times[k] * 1e3))
    end
end
# Rule of thumb: always discard the 1st call, then measure the steady state
# (exactly what @btime does). What IS this compilation? Next section.

#%% 3. Interpreted vs compiled — observe the compilation
# Three ways from source to processor:
#   - interpreted (CPython): bytecode re-examined at every op → slow, no warmup;
#   - AOT (C, numpy): compiled before running → fast, no warmup;
#   - JIT (Julia): compiled on the 1st call, then native → fast AFTER the warmup.
# Julia lets you INSPECT every stage. A deliberately trivial function keeps it readable.
f(x) = 2x + 1

#%% Step 1 — the lowered code (canonical SSA form, still type-independent)
@code_lowered f(3.0)

#%% Step 2 — the typed code (the DECISIVE step)
# Julia now knows the argument type and propagates types through the function.
# This is where "one compilation per type" happens: f(3.0) (Float64) and f(3) (Int)
# trigger TWO distinct compiled versions (a "specialization" / method instance),
# built on first encounter of each type combination, then reused. Compare:
@code_typed f(3.0)

#%% ...and for an Int argument
@code_typed f(3)

#%% Step 3 — the assembly (per-type compilation becomes directly observable)
# f(3.0) (Float64) → floating-point instructions (vmulsd, vaddsd…)
# f(3)   (Int)     → integer instructions (lea, add…)
# Two types → two different machine codes, a few lines each, no interpreter loop.
@code_native debuginfo=:none f(3.0)

#%% ...the integer version
@code_native debuginfo=:none f(3)

#%% What to remember (interpreted vs compiled)
# - Python (CPython) keeps code as bytecode and runs it through an interpretation
#   loop, on BOXED objects allocated on the heap. That per-operation cost
#   penalizes the loop.
# - Julia compiles, on first use and per type combination, a specialized native
#   version. Once compiled, 2x+1 on Float64 is literally two machine instructions.
# "Julia is compiled" = just-in-time (JIT) compilation, triggered by argument types.
#
# ⚠ DON'T say "Python can't specialize by type" — it's false since 3.11 (PEP 659),
# and the true story is a BETTER argument. CPython's adaptive interpreter watches
# the types flowing through each bytecode and rewrites it in place: a warmed-up
# `s += x` on floats really becomes BINARY_OP_ADD_FLOAT, no __add__ lookup left.
# (dis.dis() hides it — it defaults to adaptive=False. Use
#  dis.get_instructions(f, adaptive=True).)
# The difference is WHAT is specialized and WHAT survives:
#   CPython  : ONE bytecode, guard RE-CHECKED every execution, still in the
#              interpreter loop, still boxed. It removed the method lookup — the
#              cheapest part.
#   Julia    : the WHOLE function, guard PROVED once and deleted, native code,
#              unboxed values in registers.
# That's why the ~100× survives PEP 659. The Python track measures this (python/1).
#
# KEY DISTINCTION — what is specialized, and does the check survive?
#   - Python: one instruction, guard re-checked at every operation, boxed → slow.
#   - Julia/JIT: the whole function, known once, the type check disappears from
#     the hot code, unboxed → fast.
# This compile-time vs run-time opposition returns for dispatch, threads and GPU.

#%% Going further: why Base `sum` beats our loop
# mysum is already native. Yet Base `sum` is faster — same computation. To
# isolate the COMPUTATION (not memory), take a vector that fits in cache:
const vc = rand(100_000)   # ~800 KB: fits in cache
@btime mysum($vc)

#%% ...and Base sum on the same data
@btime sum($vc)

#%% Our loop's assembly
# Base `sum` is ~5× faster; the reason is in the assembly. Ours shows `vaddsd`:
# Scalar Double = ONE float at a time, in a SINGLE accumulator (xmm0). Each `+`
# waits for the previous → one dependency chain (latency-bound).
@code_native debuginfo=:none mysum(vc)

#%% Now `sum`'s assembly — and the step everyone skips
# ⚠ Look closely: NO vaddpd, NO ymm here either — just a handful of vaddsd. If you
# stop reading now you'll conclude `sum` isn't vectorized at all. It is; the work
# just ISN'T IN THIS FUNCTION. Three things to spot in the listing:
#
#     cmp     rdx, 15
#     jle     .LBB0_6                        ← N ≤ 15? handle it right here
#     movabs  rax, offset j_mapreduce_impl   ← otherwise CALL the real kernel
#     mov     ecx, 1024                      ← ...with a block size of 1024
#     call    rax
#
# So the vaddsd you DO see belong to the small-N path (N ≤ 15, fully unrolled) —
# not to the hot loop. The N ≥ 16 threshold and the 1024-element pairwise block are
# literally readable in the machine code (they live in Base's reduce.jl).
@code_native debuginfo=:none sum(vc)

#%% ...and NOW the kernel that does the work
# Follow the call. THIS is where the SIMD lives:
@code_native debuginfo=:none Base.mapreduce_impl(identity, Base.add_sum, vc, 1, length(vc), 1024)
# You'll find `vaddpd ymm0..ymm3`: Packed Double = 4 floats at once (SIMD) × 4
# INDEPENDENT accumulators → ~16 doubles/iteration, 4 dependency chains in parallel
# (ILP). Then they're folded together: vaddpd ymm0,ymm1,ymm0 / ymm2 / ymm3.
#
# Why can't OUR loop do that? Float addition is NOT associative ((a+b)+c ≠ a+(b+c)
# via rounding): without permission the compiler MUST keep our order. `sum` goes
# through a reduction that ALLOWS reassociation. We grant the same permission with
# @simd:

#%% The same loop, with @simd
function mysum_simd(x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)   # @simd: allows reassociation
        s += x[i]
    end
    return s
end
@btime mysum_simd($vc)
# One word (@simd) and the loop vectorizes: it catches up with (or beats) `sum`.
# Speed isn't a "magic" language but WHAT THE COMPILER IS ALLOWED TO DO — here,
# reassociate to exploit SIMD + instruction-level parallelism.
# (Reproducible: experiences/sum_vs_size/reflection.jl)

#%% 4. Pitfall #1: type instability
# "Julia is fast" is false if the compiler CAN'T determine the types. Most common
# case: an untyped global variable. Same sum, but reading a global.
glob = rand(1000)   # global variable (type not fixed from the compiler's view)

function sum_global()
    s = 0.0
    for i in eachindex(glob)   # glob is a global
        s += glob[i]
    end
    return s
end

function sum_arg(x)
    s = 0.0
    for i in eachindex(x)
        s += x[i]
    end
    return s
end
(sum_global(), sum_arg(glob))

#%% @code_warntype: red = the type is unknown (Any) → dynamic fallback, like Python
# The global version (expect red / Any):
@code_warntype sum_global()

#%% ...the version taking its array AS AN ARGUMENT (everything typed, no red):
@code_warntype sum_arg(glob)

#%% The lesson, measured — same computation, measure the gap
@benchmark sum_global()

#%% ...vs the argument version
@benchmark sum_arg($glob)
# JULIA GOLDEN RULE: put the work in FUNCTIONS that receive their data AS
# ARGUMENTS. That's what enables type specialization — hence speed. A loop "at
# global scope" is a common mistake when coming from Python.

#%% 5. Why numpy is fast (and what it hides)
# numpy.sum(x) is ~100× faster than a Python loop because it DOESN'T run the loop
# in Python: contiguous typed array + a compiled C function. Python only *calls* it.
# Consequence:
#   - Stay in vectorized numpy ops (x + y, x.sum(), x @ y) → fast.
#   - Write a Python loop around numpy elements → interpreter cost per iteration,
#     and it all collapses.
# Julia has no such boundary: loop AND "vectorized" form are the same compiled
# language. That's the two-language problem — Python writes fast parts in C/Cython/
# Numba; Julia tries to eliminate it.
# Numba: @njit adds a JIT to a SUBSET of Python — Julia's mechanism bolted on for a
# restricted domain. "JIT" isn't magic: specialize by types, then compile.

#%% 6. Measuring properly — the art of benchmarking
# A wrong number is worse than no number. Two pitfalls:
#   A — timing the compilation (the warmup, section 2): always discard the 1st call.
#   B — measuring only once: a single measurement is polluted by noise (OS, CPU
#       frequency, cache). Repeat and look at the distribution — the point of
#       BenchmarkTools. @btime shows the MINIMUM + allocations; @benchmark the full
#       distribution. Always interpolate variables with $ ( @btime f($x) ) so they
#       aren't treated as globals — else you measure the section-4 pitfall.
@benchmark mysum($v)

#%% 7. Profiling — WHERE does the time actually go?
# @btime answers "how long?". On a real script with several functions the question is
# "WHERE?" — and you cannot @btime every line. `Profile` (stdlib) is a SAMPLING
# profiler: it interrupts the program every few ms and records the call stack. The
# cost is negligible and nothing needs instrumenting; a function's sample count is
# roughly its share of the time.
#
# ⚠ Run this section with ONE thread (julia --threads=1, the default). With several
#   threads, the idle ones flood the report with poptask/wait frames (we come back to
#   that in the threads module, where that noise becomes the actual lesson).

# A small layered pipeline. Which of the three costs the most? Bet before you look.
clean(v)     = [x for x in v if x > 0.01]      # keep the useful values
transform(v) = sqrt.(abs.(v))
function score(v)                               # sin+cos+exp in a loop — surely THIS one?
    s = 0.0
    for x in v
        s += sin(x) * cos(x) * exp(-x)
    end
    return s
end
pipeline(v) = score(transform(clean(v)))

const pdata = rand(3_000_000)
pipeline(pdata)          # warm up first — never profile the compilation

#%% Profile it and read the flat report
Profile.clear()
Profile.@profile for _ in 1:10
    pipeline(pdata)
end
Profile.print(format = :flat, sortedby = :count, mincount = 60)

#%% What the report says
# Typical counts (the absolute numbers vary run to run — the ORDER is the lesson):
#
#     620  clean            ← MORE than score. Nobody bets on this one.
#     602  push!             ⎫
#     543  _growend!         ⎬ all of it UNDER clean: it isn't computing, it's ALLOCATING
#     524  GenericMemory     ⎭
#     490  score(v)         ← the "obvious" suspect, only second
#     157  transform(v)
#
# `[x for x in v if cond]` cannot know the final size, so it GROWS the array:
# reallocate + copy, again and again. The profiler pointed straight at a one-line
# function that looks free — and told us the cost is memory, not arithmetic.

#%% Fix what the profiler found, then measure again
clean_fast(v) = filter(>(0.01), v)   # allocates the full size ONCE, then resizes down
@assert clean(pdata) == clean_fast(pdata)
print("comprehension : "); @btime clean($pdata)
print("filter        : "); @btime clean_fast($pdata)
# comprehension : ~7 ms  (36 allocations: 80.48 MiB)
# filter        : ~4 ms  ( 3 allocations: 22.89 MiB)
# → ~1.7x faster and 3.5x less memory, for one word changed.

#%% "So it was all about the growth?" — test that hypothesis, don't assume it
# If regrowing the array were the whole story, then pre-sizing it should recover ALL
# the lost time. Let's write exactly that: same push! loop, but told the size upfront.
function clean_sizehint(v)
    out = similar(v, 0)
    sizehint!(out, length(v))         # no more regrowing: reserve everything now
    for x in v
        x > 0.01 && push!(out, x)
    end
    out
end
@assert clean_sizehint(pdata) == clean_fast(pdata)
print("push!+sizehint! : "); @btime clean_sizehint($pdata)
# → 3 allocations, 22.89 MiB: EXACTLY filter's allocations. And yet it is still
#   clearly slower than filter. So growth was NOT the whole story.
#
# WHY? Read Base's filter (array.jl:2932) — it is worth the detour:
#
#     function filter(f, a::Array{T, N}) where {T, N}
#         j = 1
#         b = Vector{T}(undef, length(a))
#         for ai in a
#             @inbounds b[j] = ai                      # write ALWAYS, unconditionally
#             j = ifelse(f(ai)::Bool, j+1, j)          # only the CURSOR is conditional
#         end
#         resize!(b, j-1); sizehint!(b, length(b)); b
#     end
#
# It has NO BRANCH in the loop. It writes every element, then decides whether to keep
# it by moving j — with `ifelse`, which compiles to a conditional move, not a jump.
# Our push! loop branches on every element; with random data the predictor is wrong a
# good fraction of the time, and each miss costs ~15-20 cycles of pipeline flush.
#
# So the honest accounting is:
#   - the GROWTH explains the MEMORY (80.48 → 22.89 MiB) and part of the time;
#   - the rest is the BRANCH — same allocations, still slower.
# Which is the module's thesis, one level down: speed is WHAT THE MACHINE MUST DO PER
# ELEMENT. Not the language, not even the allocations alone.

#%% The loop that matters
# @btime says HOW LONG · the profiler says WHERE · the allocations say WHY.
# profile → diagnose → fix → re-measure. That is the method of this whole course —
# and what you are graded on, whatever the language.

#%% 8. Application — Monte-Carlo π
# Estimate π by drawing random points in [0,1]² and counting the fraction in the
# quarter disk:   π ≈ 4 × #{x² + y² ≤ 1} / n
# We meet it again later (sequential → multi-thread → GPU).
function estimate_pi(n)
    hits = 0
    for _ in 1:n
        x = rand(); y = rand()
        if x*x + y*y <= 1.0
            hits += 1
        end
    end
    return 4 * hits / n
end
estimate_pi(10_000_000)

#%% Q1 — measure estimate_pi with @btime for n = 10⁷
# How long? How many allocations? (rand() with no argument doesn't allocate.)
@btime estimate_pi(10_000_000)

#%% Q2 — check type stability (and learn what to IGNORE)
# Run it. Body::Float64, hits::Int64 — everything concrete, NO red anywhere.
# Two things worth understanding rather than pattern-matching:
#
#  1. Why is the result always Float64, never Int, whatever the branches do?
#     Because `/` is ALWAYS true division in Julia (the Julia-basics module!):
#     4 * hits / n returns
#     Float64 even for 4 * 0 / 1000. Check: Base.return_types(estimate_pi, (Int,)).
#     The `if` changes the VALUE of hits, never its TYPE.
#
#  2. The one highlighted line is NOT a bug:
#         @_3::Union{Nothing, Tuple{Int64, Int64}}
#     That is the ITERATOR PROTOCOL: `iterate` returns either a (value, state) tuple
#     or `nothing` when exhausted. It appears in EVERY for loop you will ever write,
#     the compiler union-splits it, and it costs nothing.
#
# ⚠ Learning to IGNORE that yellow is as important as spotting the real red. Section 4
# said "red = Any = bad"; if you hunt for red here you will find this yellow instead
# and conclude that a normal for loop is type-unstable — the exact opposite of the
# lesson. Compare with sum_global() above, where `Any` really does show up.
@code_warntype estimate_pi(1000)
Base.return_types(estimate_pi, (Int,))   # → [Float64]. Always.

#%% Q3 — error decreases like 1/√n
# To gain one decimal digit you need ~100× more points. Why is this the PERFECT
# candidate for parallelism? Each draw is independent — "embarrassingly parallel"
# (keep this for the threads module). Watch the error shrink:
let
    for n in (10^4, 10^5, 10^6, 10^7, 10^8)
        err = abs(estimate_pi(n) - π)
        println(@sprintf("n = 10^%d   error = %.5f", round(Int, log10(n)), err))
    end
end

#%% 9. Wrap-up
# Interpreted (Python)  bytecode + interpretation loop → cost per operation
# JIT-compiled (Julia)  type specialization → native code, no per-op overhead
# numpy / Numba         fast because they LEAVE the Python interpreter (C / restricted JIT)
# Type stability        the condition for speed; @code_warntype, no globals
# Benchmark             @btime / @benchmark, interpolate variables
#
# TAKEAWAY: speed doesn't come from the language "in itself" but from WHAT THE
# MACHINE MUST DO PER OPERATION. Compiling + specializing removes the overhead;
# once memory is on the table, one enemy remains: the time to fetch data.
#
# WHAT'S NEXT: first Julia's multiple dispatch (vs Python's OOP); then threads and
# a classic case — the parallel sum that gives a wrong result.

#%% 10. Bonus — watch π converge (non-interactive)
# The Pluto twin has a draggable slider + live scatter plot. In a plain script we
# just print the estimate as the number of darts grows: it closes in on π, slowly,
# like 1/√n (that's Q3, seen numerically).
let
    rng = MersenneTwister(1)
    for n_pts in (100, 300, 1_000, 3_000, 10_000, 100_000)
        pts = ((rand(rng), rand(rng)) for _ in 1:n_pts)
        nin = count(p -> p[1]^2 + p[2]^2 <= 1.0, pts)
        est = 4 * nin / n_pts
        println(@sprintf("n = %6d   π ≈ %.4f   error = %.4f", n_pts, est, abs(est - π)))
    end
end
