# Concurrency vs parallelism — waiting (Part A), then computing (Part B).  ~10 min.
# Part A: download 20 days of Bitcoin prices, one by one vs all at once (real network I/O).
# Part B: count darts in the disk (Monte-Carlo π) on several threads.
# One blank per part. Run each #%% cell, read the numbers.
#
# ⚠ Part B needs SEVERAL THREADS. Start Julia with:   julia --threads=auto
#   (VS Code: set  "julia.NumThreads": "auto"  and restart the Julia REPL.)
# Needs UnicodePlots:  using Pkg; Pkg.add("UnicodePlots")

# ============================================================================
# Part A — I/O: overlap the waits
# ============================================================================
#%% setup
using Downloads, Dates, UnicodePlots
SYMBOL = "BTCUSDT"
days = [Date(2024, 6, 1) + Day(d) for d in 0:19]      # 20 days

# download ONE day's kline from Binance and return its close price (given — just read it)
function close_on(day)
    ms = round(Int, Dates.datetime2unix(DateTime(day)) * 1000)
    io = IOBuffer()
    Downloads.download(
        "https://api.binance.com/api/v3/klines?symbol=$(SYMBOL)&interval=1d&startTime=$(ms)&limit=1", io)
    m = match(r"\[\d+,\"[\d.]+\",\"[\d.]+\",\"[\d.]+\",\"([\d.]+)\"", String(take!(io)))
    return parse(Float64, m.captures[1])              # column 4 = close price
end

#%% sequential — one request, wait, next… (slow on purpose)
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

#%% Run me — the 20 requests, overlapped
t0 = time()
closes = download_all(days)
t_con = time() - t0
@assert closes == closes_seq
println("concurrent : ", round(t_con, digits=1), " s")
println("→ ~", round(Int, t_seq / t_con), "× faster — the requests WAIT on the network, and the waits overlap.")
# @async gives CONCURRENCY: while one request waits, another runs — all on ONE thread.
# In Python, plain threads do the same (the GIL is released during I/O):
# 3_concurrency_python.ipynb measures it.

#%% And here are the returns — the reason we downloaded anything at all
returns = closes[2:end] ./ closes[1:end-1] .- 1        # daily returns
cum = cumprod(1 .+ returns)                            # 1 unit invested on day 1
lineplot(cum, title="$(SYMBOL) cumulative return", xlabel="day", ylabel="× invested", height=8)

# ============================================================================
# Part B — CPU work: nobody waits. Does @async still help?
# ============================================================================
#%% setup — the dart counter from the Julia basics, and how many threads we have
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
n = 100_000_000
println("threads: ", k)

#%% Given (just run it, 2-3 times) — every task adds into the SAME counter
function pi_race(n, k)
    per = n ÷ k
    hits = 0
    @sync for _ in 1:k
        Threads.@spawn for _ in 1:per
            x, y = rand(), rand()
            hits += (x^2 + y^2 ≤ 1)       # ⚠ read-modify-write on a SHARED variable
        end
    end
    return 4 * hits / (per * k)
end
for _ in 1:3
    println("pi_race : ", pi_race(10_000_000, k), "   ← wrong, and different every run")
end
# A RACE CONDITION: two threads read the same `hits`, both add 1, one write is lost.
# It does not crash — it LIES. (With k = 1 it is right: no one to race with.)

#%% Your turn: no sharing — each task counts on its own, then we add the k results
function pi_spawn(n, k)
    per = n ÷ k
    #= SOLUTION: one Threads.@spawn count_in_disk(per) per task, then 4 * (sum of the fetch-ed counts) / (per * k) =#
    tasks = [Threads.@spawn count_in_disk(per) for _ in 1:k]
    return 4 * sum(fetch, tasks) / (per * k)
    #= END =#
end

#%% Run me — correct now; and faster?
println("pi_spawn = ", pi_spawn(n, k), "   (π = ", Float64(π), ")")   # also compiles it
count_in_disk(1000)                                     # compile before timing
t_seq   = @elapsed count_in_disk(n)
t_spawn = @elapsed pi_spawn(n, k)
println("one thread : ", round(t_seq, digits=3), " s")
println("@spawn × $k : ", round(t_spawn, digits=3), " s   → ", round(t_seq / t_spawn, digits=1), "×")

#%% Given — the SAME code with @async instead of Threads.@spawn. Predict before running.
function pi_async(n, k)
    per = n ÷ k
    tasks = [@async count_in_disk(per) for _ in 1:k]
    return 4 * sum(fetch, tasks) / (per * k)
end
pi_async(1000, k)
t_async = @elapsed pi_async(n, k)
println("@async × $k : ", round(t_async, digits=3), " s   → ", round(t_seq / t_async, digits=1), "×")
# Correct, but ~1×: @async tasks take turns on ONE thread. In Part A that was a huge win —
# there they all WAITED. Here nobody waits, so there is nothing to overlap.
# CONCURRENCY (@async) overlaps waits; PARALLELISM (@spawn) uses several cores. One word apart.
# (In Python, threads can't do Part B: the GIL serializes CPU work — you'd need processes.)
