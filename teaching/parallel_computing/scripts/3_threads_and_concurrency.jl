# =============================================================================
# Threads & concurrency — tasks, execution resources and local reductions
# ⚠ KEEP IN SYNC with `pluto/3_threads_and_concurrency.jl` (same code, two formats).
# Cells: `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools. COURSE_LIVE_IO=0 uses labelled simulated waits for Part A.
# =============================================================================

#%% 0. Getting started
using Downloads
using Base.Threads
using BenchmarkTools
println("Julia worker threads = $(nthreads())")   # 1? relaunch with julia --threads=auto (VS Code: "julia.NumThreads": "auto", restart REPL)

#%% 1. Part A — overlap waiting
println(get(ENV, "COURSE_LIVE_IO", "1") == "0" ? "SIMULATED waits + synthetic values" : "LIVE API prices")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT"]

function fetch_price(sym)
    if get(ENV, "COURSE_LIVE_IO", "1") == "0"
        sleep(0.05)  # controlled waiting, not Internet performance
        return 100.0 + something(findfirst(==(sym), SYMBOLS), 0)
    end
    url = "https://api.binance.com/api/v3/ticker/price?symbol=$(sym)"
    io = IOBuffer()
    Downloads.download(url, io)
    m = match(r"\"price\":\"([0-9.]+)\"", String(take!(io)))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end
fetch_price(SYMBOLS[1])   # warm-up: compile before timing

#%% Three ways to fetch (@elapsed once: don't hammer the API)
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

#%% 2. Part B — CPU tasks and exact batch sizes
function batch_sizes(n, ntasks)
    q, r = divrem(n, ntasks)               # q draws per batch, r left over
    sizes = fill(q, ntasks)
    sizes[1:r] .+= 1                       # the first r batches take one more
    return sizes
end

function count_in_disk(n)
    hits = 0
    for _ in 1:n
        x = rand(); y = rand()
        hits += (x * x + y * y <= 1.0)
    end
    return hits
end

pi_seq(n) = 4 * count_in_disk(n) / n

#%% Same CPU work with @async
function pi_async(n; ntasks=nthreads())
    sizes = batch_sizes(n, ntasks)
    tasks = [@async count_in_disk(size) for size in sizes]
    return 4 * sum(fetch, tasks) / n
end

let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_async = @belapsed pi_async($n)
    println("π: seq $(round(t_seq, digits=3)) s | @async × $(nthreads()): $(round(t_seq / t_async, digits=1))x")
end

#%% Shared counter — predict the estimate
function pi_race(n; ntasks=nthreads())
    hits = 0
    @sync for size in batch_sizes(n, ntasks)
        Threads.@spawn for _ in 1:size
            x = rand(); y = rand()
            hits += (x * x + y * y <= 1.0)  # intentionally shared: read–modify–write race
        end
    end
    return 4 * hits / n
end

nthreads() == 1 && @warn "1 thread: no race to see — restart with --threads=auto"
println("pi_race  : ", [round(pi_race(20_000_000), digits=5) for _ in 1:4])

#%% Return local counts, then reduce
function pi_spawn(n; ntasks=nthreads())
    sizes = batch_sizes(n, ntasks)
    tasks = [Threads.@spawn count_in_disk(size) for size in sizes]
    return 4 * sum(fetch, tasks) / n
end

println("pi_spawn : ", [round(pi_spawn(20_000_000), digits=5) for _ in 1:4])

#%% Atomic updates or local counts — measure contention
function pi_atomic(n; ntasks=nthreads())
    hits = Threads.Atomic{Int}(0)
    @sync for size in batch_sizes(n, ntasks)
        Threads.@spawn for _ in 1:size
            x = rand(); y = rand()
            (x * x + y * y <= 1.0) && Threads.atomic_add!(hits, 1)
        end
    end
    return 4 * hits[] / n
end

let n = 40_000_000
    t_seq = @belapsed pi_seq($n)
    t_atomic = @belapsed pi_atomic($n)
    println("pi_atomic = $(round(pi_atomic(n), digits=5)) | $(round(t_atomic / t_seq, digits=1))x the time of pi_seq")
end

#%% Speedup vs nthreads (relaunch with --threads=1, 2, 4)
let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_spawn = @belapsed pi_spawn($n)
    println("π: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x  (tasks=$(nthreads()), Julia worker threads=$(nthreads()))")
end

#%% 3. Optional — another split: independent Markov chains
function simulate_markov(nsteps; p_stay=0.9)
    state = 1
    count1 = 0
    for _ in 1:nsteps
        if rand() > p_stay
            state = 3 - state                # flip 1 <-> 2
        end
        count1 += (state == 1)
    end
    return count1 / nsteps
end

markov_seq(K, L) = [simulate_markov(L) for _ in 1:K]
markov_spawn(K, L) = fetch.([Threads.@spawn simulate_markov(L) for _ in 1:K])

#%% Speedup on 8 chains
let
    K, L = 8, 20_000_000
    t_seq = @belapsed markov_seq($K, $L)
    t_spawn = @belapsed markov_spawn($K, $L)
    println("markov: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x")
end
