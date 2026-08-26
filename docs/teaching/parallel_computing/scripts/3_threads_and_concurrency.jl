# =============================================================================
# Threads & concurrency — async on one core, then real multithreading
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC with `pluto/3_threads_and_concurrency.jl` — same lesson, two formats.
# ⚠ RUN WITH SEVERAL THREADS — Part B needs them:  julia --threads=auto <file>
#   (in VS Code: set "julia.NumThreads": "auto" and restart the Julia REPL).
# Cells are delimited by `#%%`. Needs BenchmarkTools; Part A calls the Binance API.
#
#   PART A — CONCURRENCY on ONE core: interleave tasks to overlap WAITING (I/O).
#   PART B — PARALLELISM on MANY cores: @spawn runs CPU work truly simultaneously.
# =============================================================================

#%% Setup
using Downloads          # stdlib HTTP client
using Base.Threads       # @spawn, nthreads
using Profile            # @profile — revisited here on threaded code
using Random             # for Q4 (the task-local RNG)
using BenchmarkTools
println("threads available: ", nthreads(), "   (Part B wants > 1)")

# ============================================================================
# PART A — asynchronous programming on a single core
# ============================================================================
# On ONE core, @async lets tasks take turns: while one WAITS on the network, another
# runs. Nothing computes twice as fast — we stop wasting the waiting time.
#   @async expr   → wrap in a Task, schedule it, return immediately
#   @sync  block  → wait for every @async task created inside
#
# 🐍 You have written this already:  await asyncio.gather(*[fetch(s) for s in syms])
#    `gather` IS @sync + @async in one call. (Or ThreadPoolExecutor().map.)
# ⚠ The GIL does NOT make Python threads useless here: CPython RELEASES it while a
#   thread waits on I/O. The GIL only blocks simultaneous BYTECODE — it kills CPU
#   parallelism, not I/O concurrency. "Threads are pointless in Python" is only true
#   for CPU work (Part B).

#%% A data source worth waiting for: Binance spot prices
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT"]

function fetch_price(sym)
    url = "https://api.binance.com/api/v3/ticker/price?symbol=$(sym)"
    io = IOBuffer()
    Downloads.download(url, io)                 # blocks THIS task, yields to others
    m = match(r"\"price\":\"([0-9.]+)\"", String(take!(io)))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end

#%% Three ways to fetch — sequential vs (wrong) async vs (real) async
# @elapsed ONCE each — never @btime here, it would hammer the API.
n = length(SYMBOLS)

# 1) SEQUENTIAL — request, wait, next request… ≈ the sum of the latencies.
seq = Vector{Float64}(undef, n)
t_seq = @elapsed for i in 1:n
    seq[i] = fetch_price(SYMBOLS[i])
end
println("1) sequential          : $(round(t_seq, digits=3)) s")

# 2) WRONG async — @async with NO @sync: we time TASK CREATION, not the fetches.
bad = fill(NaN, n)                          # NaN = "not fetched yet"
t_bad = @elapsed for i in 1:n
    @async bad[i] = fetch_price(SYMBOLS[i])
end
ready = count(!isnan, bad)
println("2) async WITHOUT @sync : $(round(t_bad, digits=5)) s   ← only task creation! ($ready/$n results ready)")

# The tasks did NOT vanish — they are still in flight. Same array, 2 s later:
sleep(2)
println("   ...2 s later, same array: $(count(!isnan, bad))/$n ready   ← they were running all along!")
# @async is EAGER: the fetches really happened, we just measured too early.
# 🐍 In Python an un-awaited coroutine NEVER starts — it stays 0/n forever. Julia
#    forgot to WAIT; Python forgot to START. Same symptom, opposite cause (python/3).
#    (The sleep also stops those orphans hogging the network during case 3.)

# 3) REAL async — @sync waits; the n waits OVERLAP → ≈ ONE round-trip.
good = Vector{Float64}(undef, n)
t_good = @elapsed @sync for i in 1:n
    @async good[i] = fetch_price(SYMBOLS[i])
end
println("3) async WITH @sync    : $(round(t_good, digits=3)) s   ← ~one round-trip, speedup $(round(t_seq / t_good, digits=1))x")
println("   e.g. $(SYMBOLS[1]) = $(round(good[1], digits=2)) USDT")
# Async overlaps WAITING, not computing. CPU-bound work on one core gains nothing.

# ============================================================================
# PART B — real multithreading with @spawn
# ============================================================================
# `Threads.@spawn f()` returns a Task the scheduler may place on ANOTHER thread;
# `fetch(task)` waits for its result. Python's threading cannot do this — the GIL
# lets one thread run bytecode at a time, so CPU parallelism needs processes.
# Julia has no GIL. This only helps if julia was started with several threads.

#%% Independent Markov chains — CPU-bound, no sharing
# 2-state chain: stay with prob p_stay, else flip. Each chain is independent →
# embarrassingly parallel.
function simulate_markov(nsteps; p_stay=0.9)
    state = 1
    count1 = 0
    for _ in 1:nsteps
        rand() > p_stay && (state = 3 - state)   # flip 1<->2
        count1 += (state == 1)
    end
    return count1 / nsteps
end

markov_seq(K, L) = [simulate_markov(L) for _ in 1:K]                 # one after another
markov_spawn(K, L) = fetch.([Threads.@spawn simulate_markov(L) for _ in 1:K])  # all at once

#%% Time them (K chains of L steps)
let
    K, L = 8, 20_000_000
    t_seq = @belapsed markov_seq($K, $L)
    t_spawn = @belapsed markov_spawn($K, $L)
    println("markov: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x  (nthreads=$(nthreads()))")
end

#%% The red thread — parallel Monte-Carlo π, and the race that ruins it
# π ≈ 4 × #{x²+y² ≤ 1} / n. Perfect for many cores — but WHERE the tasks write matters.
pi_chunk(m) = begin
    hits = 0
    for _ in 1:m
        x = rand();
        y = rand()
        hits += (x * x + y * y <= 1.0)
    end
    hits
end

pi_seq(n) = 4 * pi_chunk(n) / n

# WRONG — every task mutates one CAPTURED counter. `hits += 1` is read-add-write, not
# atomic: increments are lost. Too small AND different every run. No error, just wrong.
function pi_race(n; ntasks=nthreads())
    per = n ÷ ntasks
    hits = 0
    @sync for _ in 1:ntasks
        Threads.@spawn for _ in 1:per
            x = rand();
            y = rand()
            hits += (x * x + y * y <= 1.0)      # ⚠ shared → race condition
        end
    end
    return 4 * hits / (per * ntasks)
end

# RIGHT — each task returns its OWN count, we reduce with sum(fetch, tasks). No shared
# mutable state in the hot loop. (Better than indexing by threadid(), now discouraged.)
function pi_spawn(n; ntasks=nthreads())
    per = n ÷ ntasks
    tasks = [Threads.@spawn pi_chunk(per) for _ in 1:ntasks]
    return 4 * sum(fetch, tasks) / (per * ntasks)
end

#%% Watch the race: wrong AND unstable vs right AND stable
let
    println("π reference : ", π)
    println("pi_race  (shared counter — WRONG, differs each run):")
    for _ in 1:4
        println("   $(round(pi_race(20_000_000), digits=5))")
    end
    println("pi_spawn (return + reduce — correct, stable):")
    for _ in 1:4
        println("   $(round(pi_spawn(20_000_000), digits=5))")
    end
end

#%% Speed of the correct versions
let
    t_seq = @belapsed pi_seq(40_000_000)
    t_spawn = @belapsed pi_spawn(40_000_000)
    println("π: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x  (nthreads=$(nthreads()))")
end
# Speedup < nthreads: AMDAHL. Splitting, spawning and the final reduce stay
# sequential, and rand() has a per-task cost. Perfect scaling is a myth.

#%% Profiling threaded code — the profiler answers a DIFFERENT question here
Profile.clear()
Profile.@profile pi_spawn(200_000_000)
Profile.print(format=:flat, sortedby=:count, mincount=100, maxdepth=8)

#%% How to read that report
#     247  poptask          ⎫  the IDLE threads waiting for work — NOT your
#     247  wait             ⎬  computation. pi_chunk barely appears.
#     236  task_done_hook   ⎭
#
#     Total snapshots: 616. Utilization: 60% across all threads and tasks.
#
# THAT last line is what you read on threaded code. The question is no longer "where
# is the time?" but "ARE MY THREADS BUSY?". ~60% = 40% of the thread-time waiting:
# the sequential split/spawn/reduce (Amdahl), plus threads that finish early and idle.
# It is also why the compilation module said to profile with `--threads=1` — those
# poptask/wait frames drown a sequential profile. Same tool, two questions.

#%% Exercises
# Q1. CONCURRENCY ≠ PARALLELISM, measured. Swap @spawn for @async in pi_spawn:
#         @sync for _ in 1:8; @async pi_chunk(per); end
#     One word changed. BET FIRST, then @btime it.
#     (Answer: ~1×, no speedup. @async interleaves tasks on ONE thread, and there is
#     nothing to interleave in CPU work — nobody is waiting.)
#
# Q2. AMDAHL. Measure pi_spawn(100_000_000) with ntasks = 1, 2, 4, 8, nthreads() and
#     plot speedup vs ntasks. Why is it not ntasks×? Where does the missing time go?
#
# Q3. THE RACE, up close. Run pi_race a few times: wrong, and different every time.
#     (a) Fix it with `Threads.Atomic{Int}` + `atomic_add!`, then @btime against
#         pi_spawn. Now CORRECT — and slower. Why?
#     (b) Why does the course prefer "return + reduce" to "@atomic everywhere"?
#     (Answer: the atomic serializes every increment onto ONE cache line and the cores
#     fight over it — cache-line ping-pong. Reducing per-task results touches shared
#     memory once per TASK instead of once per draw. The cure for a race is not a
#     bigger lock: it is LESS SHARING.)
#
# Q4. IS `rand()` NOT SHARED STATE TOO?   ← the memory/GPU module comes back to this
#     Every task calls rand(), and a generator is a mutable object updated on every
#     draw. So why is pi_spawn correct? Investigate, don't guess:
#         Random.default_rng()
#         Random.default_rng() === fetch(Threads.@spawn Random.default_rng())
#         Random.seed!(1234); pi_spawn(4_000_000)     # 3× — same answer?
#         # then relaunch with --threads=1 and --threads=4 and compare.
#     (Answer: since 1.7 `rand()` uses TaskLocalRNG — the object is a stateless
#     SINGLETON marker (hence `===` is true), but the STATE lives inside the Task.
#     Each task draws from its own stream: no sharing, no race, no lock, no cost.
#     New tasks are seeded deterministically from the parent, so with a fixed seed, a
#     fixed ntasks and an in-order reduce, pi_spawn is REPRODUCIBLE on any thread count.
#     ⚠ That rests on task COUNT and creation order — change ntasks and you get a
#     different (equally valid) answer; reduce in COMPLETION order and floating-point
#     non-associativity moves the last digits.
#     🐍 Python makes you ask for this explicitly: `np.random.SeedSequence(seed).spawn(k)`
#     hands out k provably independent, reproducible streams — one per worker. See python/3,
#     section 4. Julia gives it to you by default; numpy does not, and an unseeded
#     `default_rng()` per worker is independent but irreproducible.
#     Where this goes: the GPU needs one independent stream for MILLIONS of threads.
#     Same design problem, three orders of magnitude up.)

#%% Wrap-up
# CONCURRENCY (Part A)  @async/@sync · one core · overlap WAITING · for I/O
# PARALLELISM  (Part B)  @spawn/fetch · many cores · real simultaneity · for CPU
#
#                 I/O-BOUND (waiting)                CPU-BOUND (computing)
#   ------------------------------------------------------------------------------
#   Python        threads WORK (GIL released         threads USELESS (the GIL) →
#                 during I/O), or asyncio            multiprocessing (processes)
#   Julia         @async / @sync (one core)          @spawn / fetch (many cores)
#
# Those four cells are why the two words exist: the enemy is not the same. Waiting is
# overlapped; computing must be split.
# RACE CONDITION   shared mutable state + parallel writes = silent corruption; the
#                  cure is MINIMIZING SHARING, not a bigger lock.
# AMDAHL           speedup ≠ nthreads; the sequential part caps the gain.
#
# WHAT'S NEXT: the memory hierarchy, why the CPU spends its time WAITING for memory,
# and how batching turns a memory-bound problem into a compute-bound one — on CPU,
# then on the GPU, where π finally goes massively parallel.
