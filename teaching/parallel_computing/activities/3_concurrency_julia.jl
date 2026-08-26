# Concurrency — downloading 20 days of prices, sequentially vs all at once.  ~5 min.
# Real network I/O: we pull 20 daily closes from Binance. Downloading them one-by-one WAITS
# 20 times; fill one blank to overlap the waits.
#
# No threads needed here — @async on ONE thread is enough for I/O (it's WAITING we overlap,
# not computing). Needs UnicodePlots:  using Pkg; Pkg.add("UnicodePlots")

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
# (In the warm-up, @async did NOTHING for a CPU sum — no waiting to overlap. Here it is ALL
#  waiting, so it's a huge win.) In Python, plain threads do the same: the GIL is released
#  during I/O. The Python twin measures it: 3_concurrency_python.ipynb.

#%% And here are the returns — the reason we downloaded anything at all
returns = closes[2:end] ./ closes[1:end-1] .- 1        # daily returns
cum = cumprod(1 .+ returns)                            # 1 unit invested on day 1
lineplot(cum, title="$(SYMBOL) cumulative return", xlabel="day", ylabel="× invested", height=8)
