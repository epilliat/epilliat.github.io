# Warm-up — sum ten million integers with your own loop.  ~4 minutes.
# Fill the function (the blank), run each #%% cell, read the two times.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
N = 10_000_000
v = collect(1:N)             # the integers 1, 2, …, N  (a real Vector, not a range)
S = N * (N + 1) ÷ 2          # the exact answer we can check against:  N(N+1)/2

#%% Your turn: write mysum — sum the elements of x with a plain loop (no `sum`)
function mysum(x)
    # ✏️ your turn: add up the elements of x with a for loop
    error("remove this line and write your code")
end

#%% Run me — your compiled loop vs Base `sum`
@assert mysum(v) == S
print("your loop : "); @btime mysum($v)
print("Base sum  : "); @btime sum($v)
# Both are fast: your loop is COMPILED. In Python this exact hand-loop is ~70-100× SLOWER
# than numpy — the interpreter tax (1_warmup_python.ipynb measures it). Same code,
# compiled instead of interpreted.
# (Going parallel comes with the threads module: 3_concurrency_julia.jl, Part B.)
