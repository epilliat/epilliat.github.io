# =============================================================================
# Interpreted vs compiled — why the same computation can be 100× faster
# ENSAI 3A — Julia as a test bench, Python as a point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC with `pluto/1_compilation_and_types.jl` — same lesson, two formats.
# Cells are delimited by `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools
# (`using Pkg; Pkg.add("BenchmarkTools")`); Random and Profile are stdlib.
#
# The thread of the whole course: why can the same computation, mathematically
# identical, be tens or hundreds of times faster depending on how it is written?
# Today: interpreted vs compiled · type specialization · measuring properly.
# You cannot parallelize code whose sequential cost you don't understand.
# =============================================================================

#%% 0. Getting started
using BenchmarkTools     # @btime, @benchmark
using InteractiveUtils   # @code_lowered/_typed/_native/_warntype
using Profile            # the sampling profiler (section 7)
using Random             # MersenneTwister (reproducible darts)
@btime 1 + 2             # the first use of a package precompiles — Julia compiles

#%% Same, with the full distribution
@benchmark 1 + 2

#%% 1. Same computation, very different times
N = 10_000_000
v = rand(N)
summary(v)

#%% A hand-written Julia loop
function mysum(x)
    s = zero(eltype(x))
    for i in eachindex(x)
        s += x[i]
    end
    return s
end
mysum(v)

#%% The Python point of comparison
#     def mysum(x):
#         s = 0.0
#         for xi in x: s += xi
#         return s
# On 10M elements: ~1 s in Python, a few MILLISECONDS in Julia — ~100× for code that
# reads the same line for line. numpy.sum is fast (~10 ms) because it does NOT run in
# Python: it calls a compiled C routine (section 5). So why is the JULIA loop fast?

#%% 2. The warmup: the first call is special
let
    g(x) = sum(abs2, x)        # new function, never compiled
    @time g(v)                 # 1st call: includes compilation
    @time g(v)                 # 2nd call: the real cost
end

#%% The warmup over several calls
# Trick: call through an abstract Function[...] container so the compilation falls
# INSIDE the first timed call.
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
        println("call $(lpad(k, 2)) : $(lpad(round(times[k] * 1e3, digits=3), 8)) ms")
    end
end
# Always discard the 1st call and measure the steady state — what @btime does.

#%% 3. Interpreted vs compiled — watch it happen
#   interpreted (CPython): bytecode re-examined at every op → slow, no warmup
#   AOT (C, numpy)       : compiled before running        → fast, no warmup
#   JIT (Julia)          : compiled on the 1st call       → fast AFTER the warmup
# Julia lets you inspect every stage. A trivial function keeps the listings readable.
f(x) = 2x + 1

#%% Step 1 — the lowered code (canonical form, still type-independent)
@code_lowered f(3.0)

#%% Step 2 — the typed code (the DECISIVE step)
# Julia now knows the argument type. f(3.0) and f(3) trigger TWO distinct compiled
# versions ("specializations"), built on first encounter and then reused.
@code_typed f(3.0)

#%% ...and for an Int argument
@code_typed f(3)

#%% Step 3 — the assembly: per-type compilation, directly visible
# Float64 → vmulsd/vaddsd · Int → lea/add. Two types, two machine codes, a few
# instructions each, no interpreter loop.
@code_native debuginfo=:none f(3.0)

#%% ...the integer version
@code_native debuginfo=:none f(3)

#%% What to remember
# Python keeps bytecode and runs it through an interpretation loop on BOXED heap
# objects — that per-operation cost is what penalizes the loop. Julia compiles, per
# type combination, a specialized native version: 2x+1 on Float64 is two instructions.
#
# ⚠ Do NOT say "Python cannot specialize by type" — false since 3.11 (PEP 659), and
#   the true story is a better argument. CPython's adaptive interpreter rewrites a
#   warmed-up `s += x` into BINARY_OP_ADD_FLOAT. The difference is what SURVIVES:
#     CPython : ONE bytecode, guard RE-CHECKED every execution, still interpreted,
#               still boxed. It removed the method lookup — the cheapest part.
#     Julia   : the WHOLE function, guard PROVED once and deleted, native, unboxed.
#   That is why the ~100× survives PEP 659. Measured in python/1.
#   (dis.dis() hides it — use dis.get_instructions(f, adaptive=True).)

#%% Going further: why Base `sum` beats our loop
# Take a vector that fits in cache, to isolate the COMPUTATION from memory:
vc = rand(100_000)   # ~800 KB
@btime mysum($vc)

#%% ...and Base sum on the same data
@btime sum($vc)

#%% Our loop's assembly
# ~5× slower, and the reason is here: `vaddsd` = Scalar Double, ONE float at a time
# into a SINGLE accumulator. Each `+` waits for the previous — one dependency chain.
@code_native debuginfo=:none mysum(vc)

#%% Now `sum`'s assembly — and the step everyone skips
# ⚠ No vaddpd here either! Don't conclude `sum` isn't vectorized — the work is simply
#   NOT IN THIS FUNCTION. The listing shows it delegating:
#       cmp rdx, 15 / jle ...            ← N ≤ 15? handle it inline
#       movabs rax, offset j_mapreduce_impl   ← otherwise CALL the real kernel
#       mov ecx, 1024                    ← ...with a pairwise block size of 1024
#   The vaddsd you see belong to the small-N path. The threshold and the block size
#   are readable straight from the machine code (they live in Base's reduce.jl).
@code_native debuginfo=:none sum(vc)

#%% ...and NOW the kernel that does the work
@code_native debuginfo=:none Base.mapreduce_impl(identity, Base.add_sum, vc, 1, length(vc), 1024)
# Here are the `vaddpd ymm0..ymm3`: Packed Double = 4 floats at once (SIMD) × 4
# INDEPENDENT accumulators ≈ 16 doubles per iteration, 4 dependency chains in
# parallel — then folded together.
#
# Why can't OUR loop do that? Float addition is NOT associative, so without
# permission the compiler must keep our order. `sum` uses a reduction that ALLOWS
# reassociation. We grant the same permission with @simd:

#%% The same loop, with @simd
function mysum_simd(x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)   # @simd: reassociation allowed
        s += x[i]
    end
    return s
end
@btime mysum_simd($vc)
# One word, and the loop vectorizes to match `sum`. Speed is not a magic language:
# it is WHAT THE COMPILER IS ALLOWED TO DO. (experiences/sum_vs_size/reflection.jl)

#%% 4. Pitfall #1: type instability
# "Julia is fast" is false when the compiler CANNOT determine the types. The classic
# case: an untyped global.
glob = rand(1000)

function sum_global()
    s = 0.0
    for i in eachindex(glob)   # reads a global
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

#%% @code_warntype: red = the type is Any → dynamic fallback, like Python
@code_warntype sum_global()

#%% ...the version taking its array AS AN ARGUMENT (all typed, no red)
@code_warntype sum_arg(glob)

#%% The same computation, measured
@benchmark sum_global()

#%% ...vs the argument version
@benchmark sum_arg($glob)
# GOLDEN RULE: put the work in FUNCTIONS that receive their data AS ARGUMENTS. That
# is what enables specialization. A loop at global scope is a classic Python reflex.

#%% 5. Why numpy is fast (and what it hides)
# numpy.sum doesn't run the loop in Python: contiguous typed array + compiled C, and
# Python only *calls* it. So vectorized ops (x + y, x.sum(), x @ y) are fast, but a
# Python loop around numpy ELEMENTS pays the interpreter per iteration and collapses.
# Julia has no such boundary — loop and "vectorized" form are the same compiled
# language. That is the two-language problem; Numba bolts Julia's mechanism onto a
# restricted subset of Python. "JIT" is not magic: specialize by type, then compile.

#%% 6. Measuring properly
# A wrong number is worse than no number. Two traps:
#   A — timing the compilation → discard the 1st call.
#   B — measuring once → noise (OS, CPU frequency, cache). Repeat, look at the
#       distribution. @btime = minimum + allocations; @benchmark = the whole thing.
# Always interpolate with $ (`@btime f($x)`), or you measure the section-4 pitfall.
@benchmark mysum($v)

#%% 7. Profiling — WHERE does the time actually go?
# @btime answers "how long?"; on a real script the question is "WHERE?", and you
# cannot @btime every line. `Profile` samples the call stack every few ms: no
# instrumentation, negligible cost, and a function's sample count ≈ its share.
#
# ⚠ Run this section with ONE thread (julia --threads=1). With several, the idle ones
#   flood the report with poptask/wait frames — that noise is the threads module's
#   actual lesson, but here it just drowns the signal.

# A layered pipeline. Which of the three costs the most? Bet before you look.
clean(v)     = [x for x in v if x > 0.01]
transform(v) = sqrt.(abs.(v))
function score(v)                               # sin+cos+exp in a loop — surely THIS?
    s = 0.0
    for x in v
        s += sin(x) * cos(x) * exp(-x)
    end
    return s
end
pipeline(v) = score(transform(clean(v)))

pdata = rand(3_000_000)
pipeline(pdata)          # warm up — never profile the compilation

#%% Profile it and read the flat report
Profile.clear()
Profile.@profile for _ in 1:10
    pipeline(pdata)
end
Profile.print(format = :flat, sortedby = :count, mincount = 60)

#%% What the report says
# Absolute counts vary; the ORDER is the lesson:
#     620  clean          ← MORE than score. Nobody bets on this one.
#     602  push!           ⎫
#     543  _growend!       ⎬ all UNDER clean: not computing — ALLOCATING
#     524  GenericMemory   ⎭
#     490  score          ← the "obvious" suspect, only second
#     157  transform
# `[x for x in v if cond]` cannot know the final size, so it GROWS the array:
# reallocate and copy, over and over.

#%% Fix what the profiler found, then measure again
clean_fast(v) = filter(>(0.01), v)   # allocates the full size ONCE, then resizes down
@assert clean(pdata) == clean_fast(pdata)
print("comprehension : "); @btime clean($pdata)
print("filter        : "); @btime clean_fast($pdata)
# comprehension : ~7 ms (36 allocations: 80.48 MiB)
# filter        : ~4 ms ( 3 allocations: 22.89 MiB)

#%% "So it was all about the growth?" — test that, don't assume it
# If regrowing were the whole story, pre-sizing should recover ALL the lost time.
function clean_sizehint(v)
    out = similar(v, 0)
    sizehint!(out, length(v))         # reserve everything up front
    for x in v
        x > 0.01 && push!(out, x)
    end
    out
end
@assert clean_sizehint(pdata) == clean_fast(pdata)
print("push!+sizehint! : "); @btime clean_sizehint($pdata)
# → 3 allocations, 22.89 MiB: EXACTLY filter's. And still clearly slower. So growth
#   was NOT the whole story. Base's filter (array.jl:2932) has NO BRANCH in its loop:
#
#       @inbounds b[j] = ai                 # write ALWAYS, unconditionally
#       j = ifelse(f(ai)::Bool, j+1, j)     # only the CURSOR is conditional
#
#   `ifelse` compiles to a conditional move, not a jump. Our push! loop branches on
#   every element; on random data the predictor misses often, ~15-20 cycles each.
# HONEST ACCOUNTING: growth explains the MEMORY and part of the time; the rest is the
# BRANCH. Same thesis one level down — speed is what the machine must do PER ELEMENT.
#
# 🐍 NOW GO LOOK AT python/1, SECTION 6 — the SAME pipeline, the OPPOSITE diagnosis. There
#    `score` dominates (2.24 s) and `clean` is nearly free (0.17 s), with 44.5 MILLION
#    function calls: the interpreter tax buries the allocations. Here the arithmetic is
#    compiled and nearly free, so the MEMORY is what sticks out. A profiler never tells you
#    what is slow in the abstract — only what dominates IN THIS LANGUAGE. That is the point
#    of running both tracks.

#%% The loop that matters
# @btime says HOW LONG · the profiler says WHERE · the allocations say WHY.
# profile → diagnose → fix → re-measure. That is the method of the whole course.

#%% 8. Application — Monte-Carlo π
# π ≈ 4 × #{x² + y² ≤ 1} / n. We meet it again: sequential → threads → GPU.
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

#%% Q1 — how long, and how many allocations? (rand() with no argument allocates none)
@btime estimate_pi(10_000_000)

#%% Q2 — check type stability, and learn what to IGNORE
# Body::Float64, hits::Int64 — all concrete, no red. Two things worth understanding:
#  1. The result is ALWAYS Float64 because `/` is always true division; the `if`
#     changes the VALUE of hits, never its TYPE.
#  2. The highlighted `@_3::Union{Nothing, Tuple{Int64,Int64}}` is NOT a bug — it is
#     the ITERATOR PROTOCOL (`iterate` returns (value, state) or nothing). It appears
#     in every for loop, gets union-split, and costs nothing.
# ⚠ Learning to ignore that yellow matters as much as spotting the real red — else
#   you conclude a plain for loop is type-unstable, the opposite of the lesson.
@code_warntype estimate_pi(1000)
Base.return_types(estimate_pi, (Int,))   # → [Float64]. Always.

#%% Q3 — the error decreases like 1/√n
# One more decimal digit costs ~100× more points. Each draw is INDEPENDENT, which is
# why this is the perfect parallel candidate — "embarrassingly parallel" (threads
# module). Watch it shrink:
let
    for n in (10^4, 10^5, 10^6, 10^7, 10^8)
        err = abs(estimate_pi(n) - π)
        println("n = 10^$(round(Int, log10(n)))   error = $(round(err, digits=5))")
    end
end

#%% 9. Wrap-up
# Interpreted (Python)  bytecode + interpretation loop → cost per operation
# JIT-compiled (Julia)  type specialization → native code, no per-op overhead
# numpy / Numba         fast because they LEAVE the Python interpreter
# Type stability        the condition for speed; @code_warntype, no globals
# Benchmark             @btime / @benchmark, interpolate with $
#
# TAKEAWAY: speed doesn't come from the language but from WHAT THE MACHINE MUST DO
# PER OPERATION. Compiling and specializing removes the overhead; once memory enters
# the picture, one enemy remains — the time to fetch the data.
#
# WHAT'S NEXT: multiple dispatch (vs Python's OOP), then threads and a classic trap —
# the parallel sum that returns the wrong answer.

#%% 10. Bonus — watch π converge (the Pluto twin has a live scatter plot)
let
    rng = MersenneTwister(1)
    for n_pts in (100, 300, 1_000, 3_000, 10_000, 100_000)
        pts = ((rand(rng), rand(rng)) for _ in 1:n_pts)
        nin = count(p -> p[1]^2 + p[2]^2 <= 1.0, pts)
        est = 4 * nin / n_pts
        println("n = $(lpad(n_pts, 6))   π ≈ $(round(est, digits=4))   error = $(round(abs(est - π), digits=4))")
    end
end
