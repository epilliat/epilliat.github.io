# Locality — Julia is column-major, so walk DOWN the columns.  ~6 minutes.
# Fill one loop, run each #%% cell, read the two times.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
A = rand(8_000, 8_000)     # 512 MB: far beyond any cache

#%% given (just read it) — sum walking ALONG ROWS: i outer, j inner
function sum_by_rows(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)       # @inbounds: skip the bounds checks
        for j in axes(M, 2)             # 2nd index innermost
            s += M[i, j]
        end
    end
    return s
end

#%% Your turn: sum walking DOWN COLUMNS — swap the two loops (j outer, i inner)
function sum_by_columns(M)
    s = zero(eltype(M))
    # ✏️ your turn: columns outside, rows inside:  for j in axes(M, 2) … for i in axes(M, 1)
    error("remove this line and write your code")
    return s
end

#%% Run me — same numbers, same result, only the walk order differs
@assert sum_by_rows(A) ≈ sum_by_columns(A)
print("along rows   (against the grain) : "); @btime sum_by_rows($A)
print("down columns (with the grain)    : "); @btime sum_by_columns($A)
# Julia stores a matrix COLUMN by column: M[i,j] and M[i+1,j] are neighbours in memory.
# Walking down a column reads contiguous memory — the cache serves whole lines at once.
# Walking along a row jumps a full column (64 kB here) every step → a cache miss per
# element. Same data, ~10× difference: LOCALITY, not arithmetic, decides the speed.
#
# numpy is the opposite (row-major) — and it HIDES this from you by silently reordering
# the iteration into memory order. A hand-written loop, in any language, does not get
# that free protection: the memory hierarchy sends the bill.
