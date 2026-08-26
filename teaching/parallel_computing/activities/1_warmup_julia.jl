# Warm-up — sum a million integers, then make it parallel.  ~12 minutes.
# Fill the functions (the blanks), run each #%% cell, read the numbers.
#
# ⚠ Part 2 needs SEVERAL THREADS. Start Julia with:   julia --threads=auto
#   (VS Code: set  "julia.NumThreads": "auto"  and restart the Julia REPL.)
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
N = 10_000_000
v = collect(1:N)             # the integers 1, 2, …, N  (a real Vector, not a range)
S = N * (N + 1) ÷ 2          # the exact answer we can check against:  N(N+1)/2

# split 1:n into k contiguous ranges (used in Part 2)
chunk_ranges(n, k) = (step = cld(n, k); [i:min(i + step - 1, n) for i in 1:step:n])

# ============================================================================
# Part 1 — the compiled loop
# ============================================================================
#%% Your turn: write mysum — sum the elements of x with a plain loop (no `sum`)
function mysum(x)
    #= SOLUTION: add up the elements of x with a for loop =#
    s = zero(eltype(x))
    for i in eachindex(x)
        @inbounds s += x[i]
    end
    return s
    #= END =#
end

#%% Run me — your compiled loop vs Base `sum`
@assert mysum(v) == S
print("your loop : "); @btime mysum($v)
print("Base sum  : "); @btime sum($v)
# Both are fast: your loop is COMPILED. In Python this exact hand-loop is ~100× SLOWER
# than numpy — the interpreter tax. Same code, compiled instead of interpreted. Module 1.

# ============================================================================
# Part 2 — now make it parallel. Three attempts; only the last one is any good.
# ============================================================================
#%% How many threads have we got?  (must be > 1 — see the header if this says 1)
Threads.nthreads()

#%% Test 1 (given, just run it) — "just add @threads" onto a shared total. Run it 2-3×.
function sum_wild(x)
    total = zero(eltype(x))
    Threads.@threads for i in eachindex(x)
        @inbounds total += x[i]      # every thread reads+writes the SAME `total`…
    end
    return total
end
println("true answer   : ", S)
for _ in 1:3
    println("wild @threads : ", sum_wild(v), "   ← wrong, and different every run!")
end
# A RACE CONDITION: `total += x[i]` is read-modify-write, and the threads stomp on each
# other, so updates are lost. It does NOT crash — it just LIES. Parallelism ≠ "add @threads".

#%% Test 2 — your turn: same work, but each task writes its OWN slot, launched with @async
function sum_async(x)
    rs = chunk_ranges(length(x), Threads.nthreads())
    parts = zeros(eltype(x), length(rs))
    @sync for (p, r) in enumerate(rs)
        #= SOLUTION: run  mysum(@view x[r])  as an @async task, storing it in parts[p] =#
        @async parts[p] = mysum(@view x[r])
        #= END =#
    end
    return sum(parts)
end

#%% Run me — correct now, but is it any faster?
@assert sum_async(v) == S
print("sequential : "); @btime mysum($v)
print("@async     : "); @btime sum_async($v)
# Correct (no shared writes), but ≈ the SAME speed. @async gives CONCURRENCY — the tasks
# take turns on ONE thread. That pays off when tasks WAIT (network, disk). Here the work
# is pure COMPUTATION: nobody waits, so there is nothing to overlap. No free lunch.

#%% Test 3 — your turn: same split, but @spawn each chunk (onto a real thread), then fetch
function sum_spawn(x)
    rs = chunk_ranges(length(x), Threads.nthreads())
    #= SOLUTION: @spawn mysum(@view x[r]) for each range r, then sum(fetch, tasks) =#
    tasks = [Threads.@spawn mysum(@view x[r]) for r in rs]
    return sum(fetch, tasks)
    #= END =#
end

#%% Run me — now it wins
@assert sum_spawn(v) == S
print("sequential : "); @btime mysum($v)
print("@spawn     : "); @btime sum_spawn($v)
# @spawn puts each task on a DIFFERENT thread → real PARALLELISM, several cores at once.
# Same split as @async, one word changed, opposite result. THAT is the whole difference
# between concurrency and parallelism — and why the wild version was wrong: sharing.
#
# (In Python, threads can't do this: the GIL serializes CPU work, so you'd reach for
#  separate processes instead. Julia has real threads — which is also what let the race
#  happen. More on all this in the threads module.)
