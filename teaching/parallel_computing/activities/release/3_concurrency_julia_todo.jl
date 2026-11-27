# Overlap the waits (Part A) and π in batches (Part B), Julia.  20 min + 30 min.
# One blank per part. Run each #%% cell; run small timings twice (the 1st run compiles).
# Part B needs several threads: julia --threads=auto  (VS Code: "julia.NumThreads": "auto", restart the REPL)

# ============================================================================
# Part A — I/O: overlap the waits
# ============================================================================
#%% setup
using Downloads, Dates
SYMBOL = "BTCUSDT"
days = [Date(2024, 6, 1) + Day(d) for d in 0:19]      # 20 days

# given: the close price of one day, from Binance
function real_close_on(day)
    ms = round(Int, Dates.datetime2unix(DateTime(day)) * 1000)
    io = IOBuffer()
    Downloads.download(
        "https://api.binance.com/api/v3/klines?symbol=$(SYMBOL)&interval=1d&startTime=$(ms)&limit=1", io; timeout=3)
    m = match(r"\[\d+,\"[\d.]+\",\"[\d.]+\",\"[\d.]+\",\"([\d.]+)\"", String(take!(io)))
    return parse(Float64, m.captures[1])              # column 4 = close price
end

# COURSE_LIVE_IO=0: simulated waits (offline)
live_io = get(ENV, "COURSE_LIVE_IO", "1") == "1"
if live_io
    try
        real_close_on(first(days))
    catch
        global live_io = false
    end
end
println(live_io ? "REAL API data" : "SIMULATED waits + synthetic values")
function close_on(day)
    if live_io
        return real_close_on(day)
    end
    sleep(0.05)
    return 100.0 + Dates.value(day - first(days))
end
# If the service fails later: live_io = false, rerun Part A from here. Part B does not need Part A.

#%% One day
@elapsed close_on(days[1])

#%% Two days, one after the other
@elapsed begin
    close_on(days[1])
    close_on(days[2])
end

#%% Two days at once
@elapsed @sync begin
    @async close_on(days[1])
    @async close_on(days[2])
end

#%% sequential — the 20 days, one after the other
t0 = time()
closes_seq = [close_on(d) for d in days]
t_seq = time() - t0
println("sequential : ", round(t_seq, digits=1), " s   (", length(days), " requests, one after another)")

#%% Your turn: fetch all 20 at once with @async, wait for them all with @sync
function download_all(days)
    out = zeros(Float64, length(days))
    @sync for (k, d) in enumerate(days)
        # ✏️ your turn: fetch close_on(d) as an @async task, storing it in out[k]
        error("remove this line and write your code")
    end
    return out
end

#%% The 20 days, overlapped
t0 = time()
closes = download_all(days)
t_con = time() - t0
@assert closes == closes_seq
println("concurrent : ", round(t_con, digits=1), " s")
println("ratio sequential / concurrent : ", round(t_seq / t_con, digits=1), "×  (", live_io ? "live I/O" : "labelled simulation", ")")

#%% Daily returns (optional)
returns = closes[2:end] ./ closes[1:end-1] .- 1        # daily returns
cum = cumprod(1 .+ returns)                            # 1 unit invested on day 1
println("Final cumulative ratio (optional interpretation): ", last(cum))

# ============================================================================
# Part B — CPU work
# ============================================================================
#%% setup — task count and execution resources are distinct
"Count how many of n random points land inside the unit quarter-disk."
function count_in_disk(n)
    inside = 0
    for _ in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 ≤ 1
            inside += 1
        end
    end
    return inside
end
function batch_sizes(n, ntasks)
    q, r = divrem(n, ntasks)               # q draws per batch, r left over
    sizes = fill(q, ntasks)
    sizes[1:r] .+= 1                       # the first r batches take one more
    return sizes
end
nbatches = 8
n = 1_000_003
println("batches: ", nbatches, "; Julia worker threads: ", Threads.nthreads(), "; draws: ", n)
@assert sum(batch_sizes(n, nbatches)) == n

#%% Given: every task updates the SAME counter (intentional race)
function pi_race(n, nbatches)
    hits = 0
    @sync for size in batch_sizes(n, nbatches)
        Threads.@spawn for _ in 1:size
            x, y = rand(), rand()
            hits += (x^2 + y^2 ≤ 1)
        end
    end
    return 4 * hits / n
end
pi_race(n, nbatches)     # run 2–3 times; explain the race, not a specific wrong value

#%% One task: schedule it, then fetch its result
t = Threads.@spawn count_in_disk(1_000_000)
fetch(t)

#%% Two tasks, two local results
t1 = Threads.@spawn count_in_disk(1_000_000)
t2 = Threads.@spawn count_in_disk(1_000_000)
fetch(t1) + fetch(t2)

#%% Your turn: each task returns a count; the parent sums
function pi_spawn(n, nbatches)
    sizes = batch_sizes(n, nbatches)
    # ✏️ your turn: spawn count_in_disk(size) for each size; fetch and sum the returned counts, then multiply by 4/n
    error("remove this line and write your code")
end

#%% Check counting logic separately from Monte-Carlo accuracy
fixed_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
@assert count(p -> p[1]^2 + p[2]^2 ≤ 1, fixed_points) == 3

#%% pi_spawn vs sequential — warm up, then repeat timings
function fastest_time(f, args...)
    f(args...)                                  # warm up (the 1st call compiles)
    times = [@elapsed(f(args...)) for _ in 1:3]
    return minimum(times)
end
println("pi_spawn = ", pi_spawn(n, nbatches), "   (π = ", Float64(π), ")")
t_seq = fastest_time(count_in_disk, n)
t_spawn = fastest_time(pi_spawn, n, nbatches)
println("sequential: ", round(t_seq, digits=3), " s; task reduction: ", round(t_spawn, digits=3), " s; speedup: ", round(t_seq/t_spawn, digits=1))

#%% Given: the same batches with @async
function pi_async(n, nbatches)
    sizes = batch_sizes(n, nbatches)
    tasks = [@async count_in_disk(size) for size in sizes]
    return 4 * sum(fetch, tasks) / n
end
t_async = fastest_time(pi_async, n, nbatches)
println("@async: ", round(t_async, digits=3), " s; speedup: ", round(t_seq/t_async, digits=1))
# The random samples are not fixed here; different estimates need not agree exactly.
# Explain time, allocations/scheduling scope, and the difference between tasks and worker threads.
