# Two loop orders (Julia)  20 min.
# Fill one loop, run each #%% cell.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
A = rand(8_000, 8_000)     # 512 MB

#%% given: i outer, j inner
function sum_by_rows(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)       # @inbounds: skip the bounds checks
        for j in axes(M, 2)             # 2nd index innermost
            s += M[i, j]
        end
    end
    return s
end

#%% Your turn: the same sum, j outer, i inner
function sum_by_columns(M)
    s = zero(eltype(M))
    # ✏️ your turn: columns outside, rows inside:  for j in axes(M, 2) … for i in axes(M, 1)
    error("remove this line and write your code")
    return s
end

#%% Rows vs columns
@assert sum_by_rows(A) ≈ sum_by_columns(A)
print("along rows   : "); @btime sum_by_rows($A)
print("down columns : "); @btime sum_by_columns($A)
