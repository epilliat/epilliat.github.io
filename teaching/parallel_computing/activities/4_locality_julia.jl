# Locality — Julia is column-major, so walk DOWN the columns.  ~6 minutes.
# Fill one loop, run each #%% cell, read the two times.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
A = rand(4000, 4000)     # a 128 MB matrix

#%% given (just read it) — sum walking ALONG ROWS: i outer, j inner
function sum_rows(A)
    s = 0.0
    for i in axes(A, 1), j in axes(A, 2)
        @inbounds s += A[i, j]
    end
    return s
end

#%% Your turn: sum walking DOWN COLUMNS — swap the loop order (j outer, i inner)
function sum_cols(A)
    s = 0.0
    #= SOLUTION: loop columns outside, rows inside:  for j in axes(A,2), i in axes(A,1) =#
    for j in axes(A, 2), i in axes(A, 1)
        @inbounds s += A[i, j]
    end
    #= END =#
    return s
end

#%% Run me — same numbers, same result, only the walk order differs
@assert sum_rows(A) ≈ sum_cols(A)
print("along rows  (against the grain) : "); @btime sum_rows($A)
print("down columns (with the grain)   : "); @btime sum_cols($A)
# Julia stores a matrix COLUMN by column: A[i,j] and A[i+1,j] are neighbours in memory.
# Walking down a column reads contiguous memory — the cache serves whole lines at once.
# Walking along a row jumps a full column (32 kB here) every step → a cache miss per
# element. Same data, several × difference: LOCALITY, not arithmetic, decides the speed.
#
# numpy is the opposite (row-major) — and it HIDES this from you by silently reordering
# the iteration into memory order. A hand-written loop, in any language, does not get
# that free protection: the memory hierarchy sends the bill.
