### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ d3000000-0000-4a00-8000-000000000001
md"""
# Threads and concurrency
### Overlap waiting, distribute computation, reduce local results

Concurrency means tasks progress over overlapping periods; parallelism means computations
execute simultaneously. A task is work to schedule, a thread executes tasks, and a core
executes instructions. These counts are not interchangeable.
Part A overlaps I/O waits; Part B distributes CPU work, identifies a shared-counter race and
returns local counts. Change COURSE_LIVE_IO to 0 for explicitly simulated waits.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000002
md"""
## 0. Getting started
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000003
begin
    using Downloads
    using Base.Threads
    using BenchmarkTools
    md"Packages loaded ✓"
end

# ╔═╡ d3000000-0000-4a00-8000-000000000004
nthreads()

# ╔═╡ d3000000-0000-4a00-8000-000000000005
md"""
For CPU parallelism start Julia with `--threads=auto`, then check `Threads.nthreads()`.
A task may execute on an available worker thread; it does not own a physical core.
The I/O demonstration can overlap waiting on a single thread.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000006
md"""
## 1. Part A — overlap waiting

`@async expr` creates and schedules a task. `@sync` waits for tasks created in its block.
Python thread pools and asyncio provide analogous submit-and-wait patterns for I/O.
This use case does not define concurrency as I/O-only or single-core-only.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000007
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT"]

# ╔═╡ d3000000-0000-4a00-8000-000000000008
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

# ╔═╡ 600c0693-c67c-4ed7-ade1-6530c79692e9
fetch_price(SYMBOLS[1])   # warm-up: compile before timing

# ╔═╡ d3000000-0000-4a00-8000-000000000009
md"""
### Three ways to fetch

Sequential · `@async` without `@sync` · `@sync` + `@async`. Timed **once** with `@elapsed` (don't hammer the API).
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000000a
function run_async_demo()
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
    return nothing
end

# ╔═╡ d3000000-0000-4a00-8000-00000000000b
try
    run_async_demo()
    get(ENV, "COURSE_LIVE_IO", "1") == "0" ? md"SIMULATED waits and synthetic values (see timings above)." : md"Live Binance API requests completed (see timings above)."
catch e
    md"⚠️ **Part A needs internet access** (Binance API). Error: `$(sprint(showerror, e))`"
end

# ╔═╡ d3000000-0000-4a00-8000-00000000000c
md"""
The unsynchronized timer measures task submission, not completed work. After yielding,
requests can complete without another write by the parent. With real I/O, completion within
two seconds is not guaranteed. An unscheduled Python coroutine is a different omission:
calling `async def` creates a coroutine that must be scheduled or awaited.
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000000d
md"""
## 3. Optional — another split: independent Markov chains

`Threads.@spawn f()` returns a task that may run on **another thread**; `fetch` waits for its result. Julia has **no GIL**.
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000000e
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

# ╔═╡ d3000000-0000-4a00-8000-00000000000f
begin
    markov_seq(K, L) = [simulate_markov(L) for _ in 1:K]
    markov_spawn(K, L) = fetch.([Threads.@spawn simulate_markov(L) for _ in 1:K])
end

# ╔═╡ c8bba89e-f531-4e2c-a426-227eb15d972f
md"""
Speedup on 8 chains.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000010
let
    K, L = 8, 20_000_000
    t_seq = @belapsed markov_seq($K, $L)
    t_spawn = @belapsed markov_spawn($K, $L)
    println("markov: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x")
end

# ╔═╡ d3000000-0000-4a00-8000-000000000011
md"""
## 2. Part B — CPU tasks and exact batch sizes

The supplied helper distributes every requested point: if n is not divisible by ntasks,
the first remainder batches receive one extra draw. Tasks are work batches; worker threads
are execution resources. A separate process-worker count is varied in the Python activity.
"""

# ╔═╡ b37c1000-0001-4000-8000-000000000001
function batch_sizes(n, ntasks)
    q, r = divrem(n, ntasks)               # q draws per batch, r left over
    sizes = fill(q, ntasks)
    sizes[1:r] .+= 1                       # the first r batches take one more
    return sizes
end

# ╔═╡ d3000000-0000-4a00-8000-000000000012
begin
    function count_in_disk(n)
        hits = 0
        for _ in 1:n
            x = rand(); y = rand()
            hits += (x * x + y * y <= 1.0)
        end
        return hits
    end

    pi_seq(n) = 4 * count_in_disk(n) / n
end

# ╔═╡ c7120db8-ccb6-451d-aee0-a64aa117f2e9
md"""
### Same CPU work with `@async`
"""

# ╔═╡ c603ac5e-eb35-4b74-88da-1171cfa44282
function pi_async(n; ntasks=nthreads())
    sizes = batch_sizes(n, ntasks)
    tasks = [@async count_in_disk(size) for size in sizes]
    return 4 * sum(fetch, tasks) / n
end

# ╔═╡ 56c666ff-5303-435f-9092-42efee54a4a7
let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_async = @belapsed pi_async($n)
    println("π: seq $(round(t_seq, digits=3)) s | @async × $(nthreads()): $(round(t_seq / t_async, digits=1))x")
end

# ╔═╡ d3000000-0000-4a00-8000-000000000015
md"""
### Shared counter: predict the estimate

The following race is intentional. Each task updates the same hits variable using a
non-atomic read–modify–write. Explain an interleaving that loses an increment, even if one
execution happens to return a plausible estimate.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000013
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

# ╔═╡ a5cfc431-9c5b-44f7-8120-2a41e41658f6
begin
    nthreads() == 1 && @warn "1 thread: no race to see — restart with --threads=auto"
    println("pi_race  : ", [round(pi_race(20_000_000), digits=5) for _ in 1:4])
end

# ╔═╡ 8b4a6129-8f25-4bff-8466-ca170fc146b5
md"""
### Return local counts, then reduce

Each task returns its local count; the parent sums the results. No counter is updated
concurrently in the reduction. The denominator is n, and the supplied sizes sum exactly to n.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000014
function pi_spawn(n; ntasks=nthreads())
    sizes = batch_sizes(n, ntasks)
    tasks = [Threads.@spawn count_in_disk(size) for size in sizes]
    return 4 * sum(fetch, tasks) / n
end

# ╔═╡ d3000000-0000-4a00-8000-000000000016
println("pi_spawn : ", [round(pi_spawn(20_000_000), digits=5) for _ in 1:4])

# ╔═╡ 9b63ddf2-0642-4ede-ac6f-c2f6ea7053f4
md"""
### Atomic updates or local counts?

`Threads.atomic_add!(hits, 1)` performs an explicit atomic increment. Merely storing the
counter in an Atomic container does not make `hits += 1` atomic. Compare one shared update
per hit with one returned count per task; measure rather than assume an atomic slowdown.
"""

# ╔═╡ e07dbcd8-4863-4996-97c4-cf5d738874d4
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

# ╔═╡ f87a0d77-71fa-46dc-81cd-c523d9c4eb7a
let n = 40_000_000
    t_seq = @belapsed pi_seq($n)
    t_atomic = @belapsed pi_atomic($n)
    println("pi_atomic = $(round(pi_atomic(n), digits=5)) | $(round(t_atomic / t_seq, digits=1))x the time of pi_seq")
end

# ╔═╡ b3e5a428-a3d4-40ed-a314-f4b6fb237ab0
md"""
### Speedup vs `nthreads`

Relaunch with `--threads=1`, `2`, `4`.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000017
let n = 100_000_000
    t_seq = @belapsed pi_seq($n)
    t_spawn = @belapsed pi_spawn($n)
    println("π: seq $(round(t_seq, digits=3)) s | @spawn $(round(t_spawn, digits=3)) s | " *
            "speedup $(round(t_seq / t_spawn, digits=1))x  (nthreads=$(nthreads()))")
end

# ╔═╡ d3000000-0000-4a00-8000-000000000019
md"""
## Choose a tool for the workload

- Waiting: overlap it with tasks, threads or asyncio as appropriate.
- Pure-Python CPU loops with the GIL enabled: compile/vectorize, or consider processes.
- Native computation: check existing library threads before adding workers.

Fixed batches and seeds preserve counts when only the Python worker count changes. Different
random samples require statistical comparison. Julia draws here are not fixed across runs.
Use Amdahl as a reference model, then test startup overhead, uneven completion times and
limited resources. More, smaller tasks can balance work but can also add overhead.
Next: memory locality and batching.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─d3000000-0000-4a00-8000-000000000001
# ╟─d3000000-0000-4a00-8000-000000000002
# ╠═d3000000-0000-4a00-8000-000000000003
# ╠═d3000000-0000-4a00-8000-000000000004
# ╟─d3000000-0000-4a00-8000-000000000005
# ╟─d3000000-0000-4a00-8000-000000000006
# ╠═d3000000-0000-4a00-8000-000000000007
# ╠═d3000000-0000-4a00-8000-000000000008
# ╠═600c0693-c67c-4ed7-ade1-6530c79692e9
# ╟─d3000000-0000-4a00-8000-000000000009
# ╠═d3000000-0000-4a00-8000-00000000000a
# ╠═d3000000-0000-4a00-8000-00000000000b
# ╟─d3000000-0000-4a00-8000-00000000000c
# ╟─d3000000-0000-4a00-8000-000000000011
# ╠═b37c1000-0001-4000-8000-000000000001
# ╠═d3000000-0000-4a00-8000-000000000012
# ╟─c7120db8-ccb6-451d-aee0-a64aa117f2e9
# ╠═c603ac5e-eb35-4b74-88da-1171cfa44282
# ╠═56c666ff-5303-435f-9092-42efee54a4a7
# ╟─d3000000-0000-4a00-8000-000000000015
# ╠═d3000000-0000-4a00-8000-000000000013
# ╠═a5cfc431-9c5b-44f7-8120-2a41e41658f6
# ╟─8b4a6129-8f25-4bff-8466-ca170fc146b5
# ╠═d3000000-0000-4a00-8000-000000000014
# ╠═d3000000-0000-4a00-8000-000000000016
# ╟─9b63ddf2-0642-4ede-ac6f-c2f6ea7053f4
# ╠═e07dbcd8-4863-4996-97c4-cf5d738874d4
# ╠═f87a0d77-71fa-46dc-81cd-c523d9c4eb7a
# ╟─b3e5a428-a3d4-40ed-a314-f4b6fb237ab0
# ╠═d3000000-0000-4a00-8000-000000000017
# ╟─d3000000-0000-4a00-8000-00000000000d
# ╠═d3000000-0000-4a00-8000-00000000000e
# ╠═d3000000-0000-4a00-8000-00000000000f
# ╟─c8bba89e-f531-4e2c-a426-227eb15d972f
# ╠═d3000000-0000-4a00-8000-000000000010
# ╟─d3000000-0000-4a00-8000-000000000019
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
