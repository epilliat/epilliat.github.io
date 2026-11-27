#!/usr/bin/env julia
# Thread scaling: one fresh Julia process per worker-thread count.
# This controls the thread-pool size. OS placement, unequal cores, load and power limits
# can still affect timings; a new process does not isolate those hardware effects.
#
# This worker measures ONE point (k = Threads.nthreads()) and appends one CSV row.
# The driver `run.sh` loops over k. Run the whole thing with:  bash run.sh
#
# Three variants of the same Monte-Carlo π (the course's running example):
#   - pi_seq    : plain sequential loop, the baseline;
#   - pi_spawn  : Threads.@spawn + fetch, return-and-reduce (real PARALLELISM);
#   - pi_async  : @sync + @async, same syntax, same work (only CONCURRENCY → ~1x).
# The third one is the point of the module: concurrency ≠ parallelism, MEASURED.
#
# Zero dependency (Base only). CSV `threads,variant,mean_s,std_s,pi_hat` in results/.

using Base.Threads
using Printf
using Random

const NPOINTS = 40_000_000     # darts per measurement
const REPS    = 5              # measurements per point
const SEED    = 0
const RESULTS = joinpath(@__DIR__, "results")

"Count darts landing in the quarter disk. Returns its OWN count — no sharing."
function pi_chunk(m)
    hits = 0
    for _ in 1:m
        x = rand(); y = rand()
        hits += (x * x + y * y <= 1.0)
    end
    return hits
end

pi_seq(n) = 4 * pi_chunk(n) / n

function batch_sizes(n, ntasks)
    n > 0 && ntasks > 0 || throw(ArgumentError("n and ntasks must be positive"))
    q, r = divrem(n, ntasks)
    return [q + (i <= r) for i in 1:ntasks]
end

"Parallel tasks: task count equals worker-thread count in this experiment."
function pi_spawn(n, ntasks)
    sizes = batch_sizes(n, ntasks)
    tasks = [Threads.@spawn pi_chunk(size) for size in sizes]
    return 4 * sum(fetch, tasks) / n
end

"Concurrent tasks: @async schedules them on their current thread."
function pi_async(n, ntasks)
    sizes = batch_sizes(n, ntasks)
    hits = zeros(Int, ntasks)
    @sync for i in 1:ntasks
        @async hits[i] = pi_chunk(sizes[i])
    end
    return 4 * sum(hits) / n
end

mean(v) = sum(v) / length(v)
std(v)  = length(v) < 2 ? 0.0 :
          sqrt(sum((x - mean(v))^2 for x in v) / (length(v) - 1))

"REPS elapsed times (s) of fn(), already compiled."
function sample_times(fn, reps)
    ts = Vector{Float64}(undef, reps)
    last = 0.0
    for k in 1:reps
        t0 = time_ns()
        last = fn()
        ts[k] = (time_ns() - t0) * 1e-9
    end
    return ts, last
end

function main()
    mkpath(RESULTS)
    Random.seed!(SEED)
    k = nthreads()

    # Compilation out of the measurement.
    pi_seq(10_000); pi_spawn(10_000, k); pi_async(10_000, k)

    csv = joinpath(RESULTS, "threads_scaling.csv")
    if !isfile(csv)
        open(csv, "w") do f
            println(f, "threads,variant,mean_s,std_s,pi_hat")
        end
    end

    variants = ("seq"   => () -> pi_seq(NPOINTS),
                "spawn" => () -> pi_spawn(NPOINTS, k),
                "async" => () -> pi_async(NPOINTS, k))

    @printf("threads = %2d  (NPOINTS = %d, REPS = %d)\n", k, NPOINTS, REPS)
    open(csv, "a") do f
        for (tag, fn) in variants
            ts, val = sample_times(fn, REPS)
            m, s = mean(ts), std(ts)
            println(f, k, ",", tag, ",", m, ",", s, ",", val)
            @printf("  %-6s %7.3f s ± %.3f   π ≈ %.5f\n", tag, m, s, val)
        end
    end
    println("  -> results/threads_scaling.csv")
end

main()
