using Downloads          # stdlib HTTP client
using Base.Threads       # @spawn, nthreads
using Profile            # @profile — the sampling profiler (stdlib)
using Random             # for Q4 (the task-local RNG)
using BenchmarkTools
println("threads available: ", nthreads(), "   (Part B wants > 1)")


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


# 3) we time all fetches, ran in parallel. notice the use of @sync and @async. @async declares a task, @sync forces to wait for all tasks to end.
good = Vector{Float64}(undef, n)
t_good = @elapsed @sync for i in 1:n
    @async good[i] = fetch_price(SYMBOLS[i])
end
println("3) async WITH @sync    : $(round(t_good, digits=3)) s   ← ~one round-trip, speedup $(round(t_seq / t_good, digits=1))x")
println("   e.g. $(SYMBOLS[1]) = $(round(good[1], digits=2)) USDT")

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

#%% Profiling threaded code — the profiler answers a DIFFERENT question
# `Profile` samples the call stack every few ms: a function's sample count ≈ its
# share of the time. On sequential code it says WHERE the time goes. On threads:
Profile.clear()
Profile.@profile pi_spawn(200_000_000)
Profile.print(format=:flat, sortedby=:count, mincount=100, maxdepth=8)

