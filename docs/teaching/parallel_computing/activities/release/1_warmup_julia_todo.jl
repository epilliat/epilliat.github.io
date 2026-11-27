# What does a loop cost? (Julia)  20 min.
# Fill the blank, run each #%% cell.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
N = 1_000_000
v = rand(N)                 # Float64, same size as the Python activity
S = sum(v)                  # reference result

#%% Your turn: mysum, a plain loop (no `sum`)
function mysum(x)
    # ✏️ your turn: add up the elements of x with a for loop
    error("remove this line and write your code")
end

#%% First call, then warmed-up call
t_first  = @elapsed mysum(v)
t_second = @elapsed mysum(v)
println("first call : ", round(t_first * 1e3, digits=2), " ms    warmed-up call : ", round(t_second * 1e3, digits=2), " ms")

#%% Your loop vs Base `sum`
@assert mysum(v) ≈ S
print("your loop : "); @btime mysum($v)
print("Base sum  : "); @btime sum($v)
