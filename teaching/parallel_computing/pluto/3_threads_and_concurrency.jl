### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ d3000000-0000-4a00-8000-000000000001
md"""
# Threads & concurrency
### Async on one core, then real multithreading

**ENSAI 3A — Julia as the test bench, Python as the point of comparison**

---

We know *why* Julia can be fast (compile and **specialize on the types**) and *how* it picks methods (**multiple dispatch**). Now: doing **several things at once**.

Two ideas, deliberately kept apart:

- **Part A — concurrency on one core.** Interleave tasks so we overlap **waiting** (network, disk). Nothing computes faster; we just stop wasting the idle time. Tool: `@async` / `@sync`.
- **Part B — parallelism on many cores.** `Threads.@spawn` runs **CPU** work truly simultaneously — and opens the door to the classic **race condition**.

> ⚙️ **Part B needs several threads.** Check the count in section 0. If it says `1`, relaunch Julia with `julia --threads=auto` before starting Pluto.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000002
md"""
## 0. How many threads do we have?
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000003
begin
    using Downloads          # stdlib HTTP client (download to an IOBuffer)
    using Base.Threads       # @spawn, nthreads
    using Printf             # @sprintf / @printf
    using Profile            # @profile — revisited here on threaded code
    using Random             # default_rng / seed! — for Q4 (the task-local RNG)
    using BenchmarkTools     # @belapsed for clean CPU timings
    md"Packages loaded ✓"
end

# ╔═╡ d3000000-0000-4a00-8000-000000000004
Threads.nthreads()

# ╔═╡ d3000000-0000-4a00-8000-000000000005
md"""
**If the cell above shows `1`**, Julia has a single thread and Part B will show no speedup. Give it several threads **before** opening Pluto:

```bash
julia --threads=auto        # as many threads as cores
```

then in that Julia session: `import Pluto; Pluto.run()`. Part A (async I/O) works with a single thread — that's exactly its point.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000006
md"""
## Part A — asynchronous programming on a single core

**Concurrency ≠ parallelism.** On **one** core, `@async` lets tasks take turns: while one task **waits** on the network, the scheduler runs another. Nothing runs twice as fast — we just stop wasting the waiting time.

This is the **producer–consumer** pattern, kept simple (**no `Channel`**): each producer task writes its **own** slot of a preallocated vector; the main task consumes the vector once every task is done.

- `@async expr` — wrap `expr` in a **task**, schedule it, return **immediately**.
- `@sync block` — **wait** for every `@async` task created inside the block.

### 🐍 The Python anchor

You have very likely written this already:

```python
import asyncio
async def fetch(s): ...
prices = await asyncio.gather(*[fetch(s) for s in symbols])
#              ^^^^^^^^^^^^^^ gather IS @sync + @async, in one call

# or, without asyncio, with a thread pool:
with ThreadPoolExecutor() as ex:
    prices = list(ex.map(fetch_price, symbols))
```

> ⚠️ **The subtlety nearly everyone gets wrong.** The **GIL does *not*** make Python threads useless here. CPython **releases** the GIL while a thread waits on I/O, so threads **do** overlap network waits. The GIL only stops threads from running Python **bytecode** simultaneously — it kills **CPU parallelism**, not **I/O concurrency**.
>
> *"Threads are pointless in Python"* is **false**: it's only true for **CPU** work — which is exactly Part B.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000007
# A data source worth waiting for: Binance spot prices (~15 symbols)
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                 "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "LTCUSDT",
                 "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT"]

# ╔═╡ d3000000-0000-4a00-8000-000000000008
# one HTTP GET; blocks THIS task while waiting, yielding to other tasks
function fetch_price(sym)
    url = "https://api.binance.com/api/v3/ticker/price?symbol=$(sym)"
    io = IOBuffer()
    Downloads.download(url, io)
    m = match(r"\"price\":\"([0-9.]+)\"", String(take!(io)))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end

# ╔═╡ d3000000-0000-4a00-8000-000000000009
md"""
### The three ways to fetch

We time each **once** with `@elapsed` — never `@btime` here (it would hammer the API):

1. **sequential** — request, wait, next… ≈ the **sum** of the latencies;
2. **wrong async** — `@async` but **no** `@sync`: the loop fires the tasks and returns instantly, so we time only **task creation**; the data isn't back yet;
3. **real async** — `@sync` + `@async`: the waits **overlap**, so the total is ≈ **one** round-trip, and the data is actually there.
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000000a
function run_async_demo()
    n = length(SYMBOLS)

    # 1) SEQUENTIAL — ≈ sum of the latencies
    seq = Vector{Float64}(undef, n)
    t_seq = @elapsed for i in 1:n
        seq[i] = fetch_price(SYMBOLS[i])
    end
    @printf("1) sequential          : %6.3f s\n", t_seq)

    # 2) WRONG async — @async but NO @sync: times only task CREATION.
    #    NaN sentinel shows the results aren't back yet.
    bad = fill(NaN, n)
    t_bad = @elapsed for i in 1:n
        @async bad[i] = fetch_price(SYMBOLS[i])
    end
    ready = count(!isnan, bad)
    @printf("2) async WITHOUT @sync  : %6.5f s   ← only task creation! (%d/%d ready)\n",
            t_bad, ready, n)

    # ...but those tasks did NOT vanish: they are STILL IN FLIGHT. Wait a moment and
    # look at the SAME array again — nobody touched it since.
    sleep(2)
    @printf("   ...2 s later, same array: %d/%d ready   ← they were running all along!\n",
            count(!isnan, bad), n)
    # (Also practical: without this wait, the orphan tasks would still be hogging the
    #  network during case 3 and make the speedup we're about to celebrate look worse.)

    # 3) REAL async — @sync waits; the n waits OVERLAP → ≈ one round-trip.
    good = Vector{Float64}(undef, n)
    t_good = @elapsed @sync for i in 1:n
        @async good[i] = fetch_price(SYMBOLS[i])
    end
    @printf("3) async WITH @sync     : %6.3f s   ← ~one round-trip, speedup %.1fx\n",
            t_good, t_seq / t_good)
    @printf("   e.g. %s = %.2f USDT\n", SYMBOLS[1], good[1])
    return nothing
end

# ╔═╡ d3000000-0000-4a00-8000-00000000000b
# run it (needs internet). The printout appears just below.
try
    run_async_demo()
    md"✓ fetched live from Binance (see the printout above)"
catch e
    md"⚠️ **Part A needs internet access** (Binance API). Error: `$(sprint(showerror, e))`"
end

# ╔═╡ d3000000-0000-4a00-8000-00000000000c
md"""
**Takeaway.** Async overlaps **waiting**, not computing. A CPU-bound loop on one core gains **nothing** from `@async`. And `@sync` is what makes the measurement *real* **and** the data *correct* — without it you just timed how fast Julia can create tasks.

### Look again at case 2 — it says more than "bad benchmark"

`0/15` ready when we measured, **`15/15` two seconds later, in the same array nobody touched since**. Those tasks were never lost: `@async` **schedules immediately**, the requests really were in flight. We simply looked before they landed.

!!! tip "🐍 The same trap in Python has the *opposite* cause — and that's the lesson"
	An un-awaited Python coroutine stays at **`0/15` forever**: `fetch_async(s)` builds an object and runs *nothing*, ever. Julia's `@async` is **eager**; Python's `async def` is **lazy**.

	> **Julia forgot to wait; Python forgot to start.**

	Same symptom (a meaningless microsecond benchmark), opposite mechanism. Confusing the two is how you write a Python program that silently does nothing — or a Julia program that races. The Python track measures both sides (`python/3`).
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000000d
md"""
## Part B — real multithreading with `@spawn`

Now we use several **cores** at once. `Threads.@spawn f()` returns a **task** the scheduler may place on **another thread**; `fetch(task)` waits for and returns its result.

> 🐍 Python's `threading` **can't** do this: the **GIL** lets only one thread run bytecode at a time, so CPU parallelism there needs *processes*. **Julia has no GIL** — which is exactly what makes real parallelism possible… and what enables the race condition below.

This only speeds things up if Julia was started with several threads (section 0).
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000000e
# A tiny 2-state Markov chain: stay with prob p_stay, else flip. Return the
# fraction of time spent in state 1. Each chain is independent → embarrassingly parallel.
function simulate_markov(nsteps; p_stay = 0.9)
    state = 1
    count1 = 0
    for _ in 1:nsteps
        rand() > p_stay && (state = 3 - state)   # flip 1<->2
        count1 += (state == 1)
    end
    return count1 / nsteps
end

# ╔═╡ d3000000-0000-4a00-8000-00000000000f
begin
    markov_seq(K, L)   = [simulate_markov(L) for _ in 1:K]                        # one after another
    markov_spawn(K, L) = fetch.([Threads.@spawn simulate_markov(L) for _ in 1:K]) # all at once
end

# ╔═╡ d3000000-0000-4a00-8000-000000000010
# K independent chains of L steps — sequential vs @spawn
let
    K, L = 8, 20_000_000
    t_seq   = @belapsed markov_seq($K, $L)
    t_spawn = @belapsed markov_spawn($K, $L)
    @printf("markov: seq %.3f s | @spawn %.3f s | speedup %.1fx  (nthreads=%d)\n",
            t_seq, t_spawn, t_seq / t_spawn, nthreads())
end

# ╔═╡ d3000000-0000-4a00-8000-000000000011
md"""
## The red thread — parallel Monte-Carlo π, and the race that ruins it

Reminder (the compilation module): π ≈ 4 × #{x² + y² ≤ 1} / n — perfect for many cores. But **where the tasks write** decides whether the answer is even correct.

First a worker returning **its own** count (no sharing), then the buggy and the correct parallel versions.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000012
begin
    # a chunk of the estimate: returns ITS OWN hit count, shares nothing
    function pi_chunk(m)
        hits = 0
        for _ in 1:m
            x = rand(); y = rand()
            hits += (x * x + y * y <= 1.0)
        end
        return hits
    end
    pi_seq(n) = 4 * pi_chunk(n) / n
end

# ╔═╡ d3000000-0000-4a00-8000-000000000013
# WRONG — spawned tasks all mutate ONE captured counter `hits` at once.
# `hits += 1` is read-add-write, not atomic → increments are lost → too small AND
# different every run. No error, just a wrong number.
function pi_race(n; ntasks = nthreads())
    per = n ÷ ntasks
    hits = 0
    @sync for _ in 1:ntasks
        Threads.@spawn for _ in 1:per
            x = rand(); y = rand()
            hits += (x * x + y * y <= 1.0)      # ⚠️ shared → race condition
        end
    end
    return 4 * hits / (per * ntasks)
end

# ╔═╡ d3000000-0000-4a00-8000-000000000014
# RIGHT — each task returns its own count; reduce with sum(fetch, tasks). No shared
# mutable state in the hot loop → correct and fast. (Replaces the older, now
# discouraged, threadid()-indexed accumulator.)
function pi_spawn(n; ntasks = nthreads())
    per = n ÷ ntasks
    tasks = [Threads.@spawn pi_chunk(per) for _ in 1:ntasks]
    return 4 * sum(fetch, tasks) / (per * ntasks)
end

# ╔═╡ d3000000-0000-4a00-8000-000000000015
md"""
### Watch the race

`pi_race` shares a counter → **wrong and unstable**. `pi_spawn` returns-and-reduces → **correct and stable**. Run it a few times:
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000016
let
    println("π reference : ", π, "\n")
    println("pi_race  (shared counter — WRONG, differs each run):")
    for _ in 1:4
        println(@sprintf("   %.5f", pi_race(20_000_000)))
    end
    println("\npi_spawn (return + reduce — correct, stable):")
    for _ in 1:4
        println(@sprintf("   %.5f", pi_spawn(20_000_000)))
    end
end

# ╔═╡ d3000000-0000-4a00-8000-000000000017
# speed of the correct versions
let
    t_seq   = @belapsed pi_seq(40_000_000)
    t_spawn = @belapsed pi_spawn(40_000_000)
    @printf("π: seq %.3f s | @spawn %.3f s | speedup %.1fx  (nthreads=%d)\n",
            t_seq, t_spawn, t_seq / t_spawn, nthreads())
end

# ╔═╡ d3000000-0000-4a00-8000-000000000018
md"""
**Amdahl.** The speedup is **less** than the thread count: splitting the work, spawning, and the final reduce stay sequential, and `rand()` has its own per-task cost. Perfect scaling is a myth — a point that becomes critical on the GPU.
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000001a
md"""
## Profiling the parallel version — the same tool, a *different* question

In the compilation module the profiler told us **where** the time went. Point it at **threaded** code and the top of the report is… the **scheduler**.
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000001b
let
    Profile.clear()
    Profile.@profile pi_spawn(200_000_000)
    Profile.print(format = :flat, sortedby = :count, mincount = 100, maxdepth = 8)
end

# ╔═╡ d3000000-0000-4a00-8000-00000000001c
md"""
### How to read that report

Typical top frames:

| count | frame | |
|---|---|---|
| 247 | `poptask` | ⎫ |
| 247 | `wait` | ⎬ the **idle threads**, waiting for work — *not* your computation |
| 236 | `task_done_hook` | ⎭ |

`pi_chunk` — the actual work — barely shows up at all. But look at the summary line:

```
Total snapshots: 616. Utilization: 60% across all threads and tasks.
```

> **That** is the line to read on threaded code. The question is no longer *"where is the time?"* but **"are my threads actually busy?"**

~60% means about **40% of the available thread-time went into waiting**: the split / spawn / reduce that stays sequential (**Amdahl**, above), plus the threads that finish early and then idle.

This is also *why* the compilation module told you to profile with `--threads=1` — those `poptask`/`wait` frames are exactly the noise that drowns a sequential profile. **Same tool, two questions: pick the right one.**
"""

# ╔═╡ d3000000-0000-4a00-8000-00000000001d
md"""
## Exercises

**Q1 — Concurrency ≠ parallelism, *measured*.** `pi_spawn` uses `@spawn`. Swap it for `@async`:

```julia
@sync for _ in 1:8
    @async pi_chunk(per)
end
```

Same syntax, same work, **one word changed**. Bet on the answer *before* you run `@btime`.

!!! hint "Answer"
	**~1×** — no speedup at all, against ~6× for `@spawn`. `@async` gives you **concurrency**: tasks interleaved on **one** thread. There is nothing to interleave in CPU work — nobody is waiting. Part A's trick buys exactly **nothing** here. That is the whole distinction of this module, in one benchmark.

**Q2 — Amdahl.** Measure `pi_spawn(100_000_000)` with `ntasks = 1, 2, 4, 8, nthreads()`. Plot speedup vs `ntasks`. Why is it never `ntasks`×? Where does the missing time go?

!!! hint "Hint"
	Re-read the **Utilization** line above, then count what stays sequential: the spawn, the `fetch`, the reduce — and the tasks that finish early and then idle.

**Q3 — The race, up close.** Run `pi_race` a few times: wrong *and* different every time.

1. Fix it with an atomic counter (`Threads.Atomic{Int}` + `atomic_add!`). `@btime` it against `pi_spawn`. It is now **correct** — and **slower**. Why?
2. So why does this course prefer *return + reduce* over *`@atomic` everywhere*?

!!! hint "Answer"
	The atomic is correct but **serializes every increment** onto a single cache line, and the cores fight over it (cache-line ping-pong). Reducing per-task results touches shared memory **once per task** instead of **once per draw**. The cure for a race is not a bigger lock: it is **less sharing**.

**Q4 — Isn't `rand()` shared state too?**  ← *the memory/GPU module comes back to this one*

`pi_chunk` calls `rand()` from every task at once. A random generator is *by definition* a mutable object that updates its state on every draw. We just spent a whole section proving that shared mutable state + parallel writes = silent corruption. **So why is `pi_spawn` correct?**

Investigate — don't guess:

```julia
Random.default_rng()                                          # what type is this?
Random.default_rng() === fetch(Threads.@spawn Random.default_rng())
Random.seed!(1234); pi_spawn(4_000_000)                       # run 3×. Same answer?
# then relaunch julia with --threads=1, then --threads=4, and compare.
```

!!! hint "Answer"
	Since 1.7, `rand()` draws from a **`TaskLocalRNG`**. The object you get back is a *stateless singleton marker* — so `===` really does return `true`, which is a nice trap — but the actual state lives **inside the `Task`**. Each task draws from **its own stream**: no sharing, no race, and **no lock either**, which is why it costs nothing.

	Better still: each new task is **seeded deterministically from its parent's stream** at creation time. So with a fixed seed, a fixed number of tasks and an in-order reduce, `pi_spawn` is **reproducible** — the same answer on 1, 4 or 22 threads, whatever the scheduler decides. Verify it. Compare with `pi_race`, which changes every run: the difference isn't luck, it's that one of them shares state and the other doesn't.

	⚠️ Don't over-claim: that reproducibility rests on the task **count** and **creation order**. Change `ntasks` and the streams are redistributed → a different (equally valid) answer. And if you reduced `Float64` partial sums in *completion* order rather than task order, floating-point non-associativity would move the last digits.

	**Where this is going:** on the GPU, "one independent stream per task" has to work for **millions** of threads at once. Same design problem, three orders of magnitude up — that's the Monte-Carlo π of the memory/GPU module.
"""

# ╔═╡ d3000000-0000-4a00-8000-000000000019
md"""
## Wrap-up

| | tool | hardware | overlaps | good for |
|---|---|---|---|---|
| **Concurrency** (A) | `@async` / `@sync` | one core | **waiting** | I/O (network, disk) |
| **Parallelism** (B) | `@spawn` / `fetch` | many cores | **computing** | CPU-bound work |

### When does what help? — both languages at once

| | **I/O-bound** (waiting) | **CPU-bound** (computing) |
|---|---|---|
| **Python** | threads **work** (the GIL is released during I/O), or `asyncio` | threads **useless** (the GIL) → `multiprocessing` |
| **Julia** | `@async` / `@sync` (one core) | `@spawn` / `fetch` (many cores) |

> Those four cells are **why "concurrency" and "parallelism" are two different words**: the enemy isn't the same. **Waiting is overlapped; computing must be split.**

- **Race condition**: shared mutable state + parallel writes = **silent corruption** (no error, just a wrong, non-deterministic result). The cure is to **minimize sharing** — return and reduce, rather than `@atomic` everywhere (which just serializes).
- **Amdahl**: speedup ≠ number of threads; the sequential part caps the gain.

### What's next

The **memory hierarchy** (L1/L2/L3/RAM), why the CPU spends its time **waiting for memory**, and how **batching** turns a memory-bound problem into a compute-bound one — on CPU, then on the **GPU**, where π finally goes massively parallel.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
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
# ╟─d3000000-0000-4a00-8000-000000000009
# ╠═d3000000-0000-4a00-8000-00000000000a
# ╠═d3000000-0000-4a00-8000-00000000000b
# ╟─d3000000-0000-4a00-8000-00000000000c
# ╟─d3000000-0000-4a00-8000-00000000000d
# ╠═d3000000-0000-4a00-8000-00000000000e
# ╠═d3000000-0000-4a00-8000-00000000000f
# ╠═d3000000-0000-4a00-8000-000000000010
# ╟─d3000000-0000-4a00-8000-000000000011
# ╠═d3000000-0000-4a00-8000-000000000012
# ╠═d3000000-0000-4a00-8000-000000000013
# ╠═d3000000-0000-4a00-8000-000000000014
# ╟─d3000000-0000-4a00-8000-000000000015
# ╠═d3000000-0000-4a00-8000-000000000016
# ╠═d3000000-0000-4a00-8000-000000000017
# ╟─d3000000-0000-4a00-8000-000000000018
# ╟─d3000000-0000-4a00-8000-00000000001a
# ╠═d3000000-0000-4a00-8000-00000000001b
# ╟─d3000000-0000-4a00-8000-00000000001c
# ╟─d3000000-0000-4a00-8000-00000000001d
# ╟─d3000000-0000-4a00-8000-000000000019
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
