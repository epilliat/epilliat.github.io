# =============================================================================
# Threads & concurrency — waiting is overlapped; computing must be split
# ⚠ KEEP IN SYNC with `pluto/3_threads_and_concurrency.jl` (same code, two formats).
# Cells: `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools and the network (Part A).
# For each cell: predict, then run.
# =============================================================================

#%% 0. Getting started
using Downloads
using Base.Threads
using BenchmarkTools
println("nthreads = $(nthreads())")   # 1? relaunch with julia --threads=auto (VS Code: "julia.NumThreads": "auto", restart REPL)

#%% 1. Part A — async on one core: waiting for the network
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT"]

function fetch_price(sym)
    url = "https://api.binance.com/api/v3/ticker/price?symbol=$(sym)"
    io = IOBuffer()
    Downloads.download(url, io)
    m = match(r"\"price\":\"([0-9.]+)\"", String(take!(io)))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end
fetch_price(SYMBOLS[1])   # warm-up: compile before timing

#%% Three ways to fetch — predict the three timings first (@elapsed once: don't hammer the API)
n = length(SYMBOLS)

seq = Vector{Float64}(undef, n)
t_seq = @elapsed for i in 1:n
    seq[i] = fetch_price(SYMBOLS[i])
end
println("1) sequential          : $(round(t_seq, digits=3)) s")

bad = fill(NaN, n)
t_bad = @elapsed for i in 1:n
    @async bad[i] = fetch_price(SYMBOLS[i])
end
println("2) @async, no @sync    : $(round(t_bad, digits=5)) s   ($(count(!isnan, bad))/$n ready)")
sleep(2)
println("   2 s later, same array: $(count(!isnan, bad))/$n ready")

good = Vector{Float64}(undef, n)
t_good = @elapsed @sync for i in 1:n
    @async good[i] = fetch_price(SYMBOLS[i])
end
println("3) @sync + @async      : $(round(t_good, digits=3)) s   speedup $(round(t_seq / t_good, digits=1))x")

#%% 2. Part B — @spawn on many cores: independent Markov chains
function simulate_markov(nsteps; p_stay=0.9)
    state = 1
    count1 = 0
    for _ in 1:nsteps
        rand() > p_stay && (state = 3 - state)   # flip 1<->2
        count1 += (state == 1)
    end
    return count1 / nsteps
end

markov_seq(K, L) = [simulate_markov(L) for _ in 1:K]
markov_spawn(K, L) = fetch.([Threads.@spawn simulate_markov(L) for _ in 1:K])

#%% Predict the speedup on 8 chains, then run
let
    K, L = 8, 20_000_000
    t_seq = @belapsed markov_seq($K, $L)
    t_spawn = @belapsed markov_spawn($K, $L)
    println("markov: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x")
end

#%% 3. Monte-Carlo π — the sequential baseline (same count as in the compilation module)
function count_in_disk(n)
    hits = 0
    for _ in 1:n
        x = rand(); y = rand()
        hits += (x * x + y * y <= 1.0)
    end
    return hits
end

pi_seq(n) = 4 * count_in_disk(n) / n

#%% Same CPU work with @async: predict the speedup
function pi_async(n; ntasks=nthreads())
    per = n ÷ ntasks
    tasks = [@async count_in_disk(per) for _ in 1:ntasks]
    return 4 * sum(fetch, tasks) / (per * ntasks)
end

let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_async = @belapsed pi_async($n)
    println("π: seq $(round(t_seq, digits=3)) s | @async × $(nthreads()): $(round(t_seq / t_async, digits=1))x")
end

#%% π on all threads, one task per thread: what does it print? Spot the bug before running
function pi_race(n; ntasks=nthreads())
    per = n ÷ ntasks
    hits = 0
    @sync for _ in 1:ntasks
        Threads.@spawn for _ in 1:per
            x = rand(); y = rand()
            hits += (x * x + y * y <= 1.0)
        end
    end
    return 4 * hits / (per * ntasks)
end

nthreads() == 1 && @warn "1 thread: no race to see — restart with --threads=auto"
println("pi_race  : ", [round(pi_race(20_000_000), digits=5) for _ in 1:4])

#%% The cure: each task returns its count, reduce once. Stable now?
function pi_spawn(n; ntasks=nthreads())
    per = n ÷ ntasks
    tasks = [Threads.@spawn count_in_disk(per) for _ in 1:ntasks]
    return 4 * sum(fetch, tasks) / (per * ntasks)
end

println("pi_spawn : ", [round(pi_spawn(20_000_000), digits=5) for _ in 1:4])

#%% Why not an atomic counter? Correct? Faster than pi_seq?
function pi_atomic(n; ntasks=nthreads())
    per = n ÷ ntasks
    hits = Threads.Atomic{Int}(0)
    @sync for _ in 1:ntasks
        Threads.@spawn for _ in 1:per
            x = rand(); y = rand()
            (x * x + y * y <= 1.0) && Threads.atomic_add!(hits, 1)
        end
    end
    return 4 * hits[] / (per * ntasks)
end

let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_atomic = @belapsed pi_atomic($n)
    println("pi_atomic = $(round(pi_atomic(n), digits=5)) | $(round(t_atomic / t_seq, digits=1))x the time of pi_seq")
end

#%% Speedup vs nthreads: predict it, then explain the gap (relaunch with --threads=1, 2, 4)
let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_spawn = @belapsed pi_spawn($n)
    println("π: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x  (nthreads=$(nthreads()))")
end
