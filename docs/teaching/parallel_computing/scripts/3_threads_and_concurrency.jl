# =============================================================================
# Threads & concurrency — async on one core, then real multithreading
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC: this is the plain-script twin of `pluto/3_threads_and_concurrency.jl`.
#   The Pluto notebook and this script hold the SAME lesson in two formats — edit
#   BOTH together whenever the content changes, so they never drift apart.
# -----------------------------------------------------------------------------
# ⚠ RUN WITH SEVERAL THREADS — Part B needs them:  julia --threads=auto <file>
#   (in VS Code: set "julia.NumThreads": "auto" and restart the Julia REPL).
# Cells are delimited by `#%%` (Alt+Enter runs a cell). Needs BenchmarkTools;
# Downloads/Printf are stdlib. Part A makes real network calls (Binance public API).
#
# Two ideas, deliberately kept apart:
#   PART A — CONCURRENCY on ONE core: interleave tasks so we overlap WAITING (I/O).
#   PART B — PARALLELISM on MANY cores: @spawn runs CPU work truly simultaneously.
# =============================================================================

#%% Setup
using Downloads          # stdlib HTTP client (download to an IOBuffer)
using Base.Threads       # @spawn, nthreads
using Printf             # @sprintf / @printf
using Profile            # @profile — revisited here on threaded code
using Random             # default_rng / seed! — for Q4 (the task-local RNG)
using BenchmarkTools     # @belapsed for clean CPU timings
println("threads available: ", nthreads(), "   (Part B wants > 1)")

# ============================================================================
# PART A — asynchronous programming on a single core
# ============================================================================
# CONCURRENCY ≠ PARALLELISM. On ONE core, @async lets tasks take turns: while one
# task WAITS on the network, the scheduler runs another. Nothing computes twice as
# fast — we just stop wasting the waiting time. This is the PRODUCER–CONSUMER
# pattern, kept simple (NO Channel): each producer task writes its OWN slot of a
# preallocated vector; the main task consumes the vector once all tasks are done.
#   @async expr   → wrap expr in a Task, schedule it, return immediately
#   @sync  block  → wait for every @async task created inside the block
#
# 🐍 THE PYTHON ANCHOR — you have very likely written this already:
#       import asyncio
#       async def fetch(s): ...
#       prices = await asyncio.gather(*[fetch(s) for s in symbols])
#                      ^^^^^^^^^^^^^^ gather IS @sync + @async in one call
#   or, without asyncio, with a thread pool:
#       with ThreadPoolExecutor() as ex:
#           prices = list(ex.map(fetch_price, symbols))
#
# ⚠ And here is the subtlety nearly everyone gets wrong: the GIL does NOT make Python
#   threads useless here. CPython RELEASES the GIL while a thread waits on I/O, so
#   threads DO overlap network waits. The GIL only stops threads from running Python
#   BYTECODE simultaneously — it kills CPU parallelism, not I/O concurrency.
#   "Threads are pointless in Python" is false: it is only true for CPU work (Part B).

#%% A data source worth waiting for: Binance spot prices
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                 "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "LTCUSDT",
                 "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT"]

function fetch_price(sym)
    url = "https://api.binance.com/api/v3/ticker/price?symbol=$(sym)"
    io = IOBuffer()
    Downloads.download(url, io)                 # blocks THIS task, yields to others
    m = match(r"\"price\":\"([0-9.]+)\"", String(take!(io)))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end

#%% The three ways to fetch — sequential vs (wrong) async vs (real) async
# We time with @elapsed, ONCE each — never @btime here (it would hammer the API).
function run_async_demo()
    n = length(SYMBOLS)

    # 1) SEQUENTIAL — one request, wait, next request, … ≈ sum of the latencies.
    seq = Vector{Float64}(undef, n)
    t_seq = @elapsed for i in 1:n
        seq[i] = fetch_price(SYMBOLS[i])
    end
    @printf("1) sequential          : %6.3f s\n", t_seq)

    # 2) WRONG async — @async but NO @sync. The loop fires n tasks and returns
    #    instantly: we time only TASK CREATION, not the fetches. The results
    #    aren't back yet — a classic bad benchmark.
    bad = fill(NaN, n)                          # NaN = "not fetched yet"
    t_bad = @elapsed for i in 1:n
        @async bad[i] = fetch_price(SYMBOLS[i])
    end
    ready = count(!isnan, bad)                   # how many actually came back?
    @printf("2) async WITHOUT @sync  : %6.5f s   ← only task creation! (%d/%d results ready)\n",
            t_bad, ready, n)

    # ...but those tasks did NOT vanish. They are STILL IN FLIGHT right now. Wait a
    # moment and look at the SAME array again — nobody touched it since:
    sleep(2)
    @printf("   ...2 s later, same array: %d/%d ready   ← they were running all along!\n",
            count(!isnan, bad), n)
    # THAT is the whole point of case 2: @async is EAGER. The fetches really happened;
    # we just measured before they landed. (🐍 Contrast with Python, where an un-awaited
    # coroutine NEVER starts — it stays 0/n forever. Julia forgot to WAIT; Python forgot
    # to START. Same symptom, opposite cause — see python/3.)
    # Practical reason for the sleep(2): if we didn't wait, these orphan tasks would
    # still be hogging the network during case 3 and make the speedup look WORSE.

    # 3) REAL async — @sync waits for all @async tasks. The n waits OVERLAP, so the
    #    total is ≈ ONE round-trip, not n. And the data is actually there.
    good = Vector{Float64}(undef, n)
    t_good = @elapsed @sync for i in 1:n
        @async good[i] = fetch_price(SYMBOLS[i])
    end
    @printf("3) async WITH @sync     : %6.3f s   ← ~one round-trip, speedup %.1fx\n",
            t_good, t_seq / t_good)
    @printf("   e.g. %s = %.2f USDT\n", SYMBOLS[1], good[1])
    return nothing
end

try
    run_async_demo()
catch e
    @warn "Part A needs internet access (Binance API) — skipping the fetch demo." exception = e
end
# Takeaway: async overlaps WAITING, not computing. CPU-bound work on one core gains
# nothing from @async. And @sync is what makes the measurement real AND the data correct.

# ============================================================================
# PART B — real multithreading with @spawn
# ============================================================================
# Now we use several CORES at once. `Threads.@spawn f()` returns a Task that the
# scheduler may place on ANOTHER thread; `fetch(task)` waits for and returns its
# result. (Python's `threading` can't do this: the GIL lets only one thread run
# bytecode at a time — real CPU parallelism there needs processes. Julia has no GIL.)
# This only speeds things up if Julia was started with several threads.

#%% Clean example — several independent Markov chains (CPU-bound, no sharing)
# A tiny 2-state chain: stay with prob p_stay, else flip. Return the fraction of
# time spent in state 1. Each chain is independent → embarrassingly parallel.
function simulate_markov(nsteps; p_stay = 0.9)
    state = 1
    count1 = 0
    for _ in 1:nsteps
        rand() > p_stay && (state = 3 - state)   # flip 1<->2
        count1 += (state == 1)
    end
    return count1 / nsteps
end

markov_seq(K, L)   = [simulate_markov(L) for _ in 1:K]                 # one after another
markov_spawn(K, L) = fetch.([Threads.@spawn simulate_markov(L) for _ in 1:K])  # all at once

#%% Time them (K chains of L steps)
let
    K, L = 8, 20_000_000
    t_seq   = @belapsed markov_seq($K, $L)
    t_spawn = @belapsed markov_spawn($K, $L)
    @printf("markov: seq %.3f s | @spawn %.3f s | speedup %.1fx  (nthreads=%d)\n",
            t_seq, t_spawn, t_seq / t_spawn, nthreads())
end

#%% The red thread — parallel Monte-Carlo π, and the race that ruins it
# Reminder (the compilation module): π ≈ 4 × #{x²+y² ≤ 1} / n. Perfect for many cores. But WHERE
# the tasks write matters. First a worker that returns ITS OWN count (no sharing):
pi_chunk(m) = begin
    hits = 0
    for _ in 1:m
        x = rand(); y = rand()
        hits += (x * x + y * y <= 1.0)
    end
    hits
end

pi_seq(n) = 4 * pi_chunk(n) / n

# WRONG — spawned tasks all mutate one CAPTURED counter `hits` at the same time.
# `hits += 1` is read-add-write, not atomic: increments are lost. Result is too
# small AND changes every run (non-deterministic) — no error, just a wrong number.
function pi_race(n; ntasks = nthreads())
    per = n ÷ ntasks
    hits = 0
    @sync for _ in 1:ntasks
        Threads.@spawn for _ in 1:per
            x = rand(); y = rand()
            hits += (x * x + y * y <= 1.0)      # ⚠ shared → race condition
        end
    end
    return 4 * hits / (per * ntasks)
end

# RIGHT — each task returns its own count; we reduce with sum(fetch, tasks). No
# shared mutable state in the hot loop → correct and fast. (This replaces the older
# `threadid()`-indexed accumulator, now discouraged.)
function pi_spawn(n; ntasks = nthreads())
    per = n ÷ ntasks
    tasks = [Threads.@spawn pi_chunk(per) for _ in 1:ntasks]
    return 4 * sum(fetch, tasks) / (per * ntasks)
end

#%% Watch the race: pi_race is wrong AND unstable; pi_spawn is right AND stable
let
    println("π reference : ", π)
    println("pi_race  (shared counter — WRONG, differs each run):")
    for _ in 1:4
        println(@sprintf("   %.5f", pi_race(20_000_000)))
    end
    println("pi_spawn (return + reduce — correct, stable):")
    for _ in 1:4
        println(@sprintf("   %.5f", pi_spawn(20_000_000)))
    end
end

#%% Speed of the correct versions
let
    t_seq   = @belapsed pi_seq(40_000_000)
    t_spawn = @belapsed pi_spawn(40_000_000)
    @printf("π: seq %.3f s | @spawn %.3f s | speedup %.1fx  (nthreads=%d)\n",
            t_seq, t_spawn, t_seq / t_spawn, nthreads())
    # Speedup < nthreads: Amdahl — splitting, spawning and the final reduce stay
    # sequential, and rand() has its own per-task cost. Perfect scaling is a myth.
end

#%% Profiling the parallel version — the profiler answers a DIFFERENT question here
# In the compilation module the profiler told us WHERE the time went. Point it at threaded code and
# the top of the report is... the scheduler:
Profile.clear()
Profile.@profile pi_spawn(200_000_000)
Profile.print(format = :flat, sortedby = :count, mincount = 100, maxdepth = 8)

#%% How to read that report
# Typical top frames:
#     247  poptask          ⎫  the IDLE threads, waiting for work —
#     247  wait             ⎬  NOT your computation.
#     236  task_done_hook   ⎭  pi_chunk barely shows up at all.
#
#     Total snapshots: 616. Utilization: 60% across all threads and tasks.
#                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# THAT is the line to read on threaded code. The question is no longer "where is the
# time?" but "ARE MY THREADS ACTUALLY BUSY?". ~60% means about 40% of the available
# thread-time went into waiting: the split/spawn/reduce that stays sequential (Amdahl,
# above), plus threads that finish early and then idle.
#
# This is also why the compilation module told you to profile with `--threads=1`: those poptask/wait
# frames are exactly the noise that drowns a sequential profile. Same tool, two
# questions — pick the right one.

#%% Exercises
# Q1. CONCURRENCY ≠ PARALLELISM, measured. pi_spawn uses @spawn. Swap it for @async:
#         @sync for _ in 1:8; @async pi_chunk(per); end
#     Same syntax, same work, one word changed. BET FIRST, then measure with @btime.
#     (Answer: ~1× vs sequential — no speedup at all. @async gives CONCURRENCY: tasks
#     interleave on ONE thread. There is nothing to interleave in CPU work — nobody is
#     waiting. Part A's trick buys exactly nothing here. That is the whole distinction,
#     in one benchmark.)
#
# Q2. AMDAHL. Measure pi_spawn(100_000_000) with ntasks = 1, 2, 4, 8, nthreads().
#     Plot speedup vs ntasks. Why is it not ntasks×? Where does the missing time go?
#     (Hint: re-read the Utilization line above, and count what stays sequential —
#     the spawn, the fetch, the reduce, and the tasks that finish early and idle.)
#
# Q3. THE RACE, up close. Run pi_race a few times: the answer is wrong AND changes.
#     (a) Fix it with an atomic counter (`Threads.Atomic{Int}` + `atomic_add!`) and
#         @btime it against pi_spawn. It is now CORRECT — and slower. Why?
#     (b) So why does the course prefer "return + reduce" over "@atomic everywhere"?
#     (Answer: the atomic is correct but serializes every increment onto one cache
#     line — the cores fight over it (false sharing / cache-line ping-pong). Reducing
#     per-task results touches shared memory ONCE per task instead of once per draw.
#     The cure for a race is not a bigger lock: it is LESS SHARING.)
#
# Q4. IS `rand()` NOT SHARED STATE TOO?  ← the memory/GPU module comes back to this
#     pi_chunk calls rand() from every task at once. A random generator is, by
#     definition, a mutable object that updates its state on every draw. We just spent
#     a whole section proving that shared mutable state + parallel writes = silent
#     corruption. So why is pi_spawn correct?
#     Investigate, don't guess:
#         Random.default_rng()                          # what type is it?
#         Random.default_rng() === fetch(Threads.@spawn Random.default_rng())
#         Random.seed!(1234); pi_spawn(4_000_000)       # run this 3×. Same answer?
#         # then relaunch julia with --threads=1 and --threads=4 and compare.
#     (Answer: since 1.7, `rand()` uses `TaskLocalRNG` — the object you get back is a
#     stateless SINGLETON marker (so `===` really is true!), but the actual state lives
#     inside the Task. Each task draws from its OWN stream: no sharing, no race, and no
#     lock either — which is why it costs nothing. Each new task is seeded
#     deterministically from its parent's stream at creation time, so with a fixed
#     seed, a fixed number of tasks and an in-order reduce, pi_spawn is REPRODUCIBLE —
#     the same answer on 1, 4 or 22 threads, whatever the scheduler does. Verify it.
#     Compare with pi_race, which changes every run: the difference is not luck, it is
#     that one of them shares state and the other does not.
#     ⚠ Don't over-claim: the reproducibility rests on task COUNT and creation order.
#     Change ntasks and the streams get redistributed → a different (equally valid)
#     answer. And if you reduced Float64 partial sums in COMPLETION order rather than
#     task order, floating-point non-associativity would move the last digits.
#     Where this is going: on the GPU, "one independent stream per task" is needed for
#     MILLIONS of threads at once. The same design problem, three orders of magnitude
#     up — that is the Monte-Carlo π of the memory/GPU module.)

#%% Wrap-up
# CONCURRENCY (Part A)   @async/@sync · one core · overlap WAITING · for I/O
# PARALLELISM  (Part B)   @spawn/fetch · many cores · real simultaneity · for CPU
#
# WHEN DOES WHAT HELP? — the table to remember, both languages at once:
#
#                 I/O-BOUND (waiting)                CPU-BOUND (computing)
#   ------------------------------------------------------------------------------
#   Python        threads WORK (the GIL is           threads USELESS (the GIL) →
#                 released during I/O), or asyncio   multiprocessing (processes)
#
#   Julia         @async / @sync (one core)          @spawn / fetch (many cores)
#
# Those four cells are WHY "concurrency" and "parallelism" are two different words:
# the enemy is not the same. Waiting is overlapped; computing must be split.
# RACE CONDITION          shared mutable state + parallel writes = silent corruption;
#                         the cure is to MINIMIZE SHARING (return + reduce, not @atomic
#                         everywhere — that just serializes).
# AMDAHL                  speedup ≠ nthreads; the sequential part caps the gain.
#
# WHAT'S NEXT: the memory hierarchy (L1/L2/L3/RAM), why the CPU spends its time
# WAITING for memory, and how batching turns a memory-bound problem into a
# compute-bound one — on CPU, then on the GPU, where π finally goes massively parallel.
