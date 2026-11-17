# Overlap the waits (Part A) and π in lots (Part B), Julia.  20 min + 30 min.
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
        #= SOLUTION: fetch close_on(d) as an @async task, storing it in out[k] =#
        @async out[k] = close_on(d)
        #= END =#
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
#%% setup
"Count how many of `n` random points land inside the unit quarter-disk."
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
k = Threads.nthreads()          # must be > 1 — see the header if this says 1
n = k * 1_000_000  # divisible by k; same total for sequential/parallel timing
println("threads: ", k)

#%% Given: every task adds into the SAME counter
function pi_race(n, k)
    per = n ÷ k
    hits = 0
    @sync for _ in 1:k
        Threads.@spawn for _ in 1:per
            x, y = rand(), rand()
            hits += (x^2 + y^2 ≤ 1)
        end
    end
    return 4 * hits / (per * k)
end
pi_race(10_000_000, k)     # run this cell 2-3 times

#%% One task on another thread: start it, then fetch its result
t = Threads.@spawn count_in_disk(1_000_000)
fetch(t)

#%% Two tasks, two results, one sum
t1 = Threads.@spawn count_in_disk(1_000_000)
t2 = Threads.@spawn count_in_disk(1_000_000)
fetch(t1) + fetch(t2)

#%% Your turn: no sharing — each task counts on its own, then we add the k results
function pi_spawn(n, k)
    per = n ÷ k
    #= SOLUTION: one Threads.@spawn count_in_disk(per) per task, then 4 * (sum of the fetch-ed counts) / (per * k) =#
    tasks = [Threads.@spawn count_in_disk(per) for _ in 1:k]
    return 4 * sum(fetch, tasks) / (per * k)
    #= END =#
end

#%% pi_spawn vs one thread
println("pi_spawn = ", pi_spawn(n, k), "   (π = ", Float64(π), ")")   # also compiles it
count_in_disk(1000)                                     # compile before timing
t_seq   = @elapsed count_in_disk(n)
t_spawn = @elapsed pi_spawn(n, k)
println("one thread : ", round(t_seq, digits=3), " s")
println("@spawn × $k : ", round(t_spawn, digits=3), " s   → ", round(t_seq / t_spawn, digits=1), "×")

#%% Given: the same code with @async
function pi_async(n, k)
    per = n ÷ k
    tasks = [@async count_in_disk(per) for _ in 1:k]
    return 4 * sum(fetch, tasks) / (per * k)
end
pi_async(1000, k)
t_async = @elapsed pi_async(n, k)
println("@async × $k : ", round(t_async, digits=3), " s   → ", round(t_seq / t_async, digits=1), "×")
