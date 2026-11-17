# What does a loop cost? (Julia)  20 min.
# Fill the blank, run each #%% cell.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
N = 10_000_000
v = collect(1:N)             # the integers 1, 2, …, N  (a real Vector, not a range)
S = N * (N + 1) ÷ 2          # the exact answer we can check against:  N(N+1)/2

#%% Your turn: mysum, a plain loop (no `sum`)
function mysum(x)
    # ✏️ your turn: add up the elements of x with a for loop
    error("remove this line and write your code")
end

#%% Try it small first
mysum([1, 2, 3])

#%% 1st call, then 2nd call
t_first  = @elapsed mysum(v)
t_second = @elapsed mysum(v)
println("1st call : ", round(t_first * 1e3, digits=2), " ms    2nd call : ", round(t_second * 1e3, digits=2), " ms")

#%% Your loop vs Base `sum`
@assert mysum(v) == S
print("your loop : "); @btime mysum($v)
print("Base sum  : "); @btime sum($v)
