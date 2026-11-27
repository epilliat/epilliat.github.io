# Two loop orders (Julia)  20 min.
# Predict before timing. Fill one compiled loop; compare sums within numerical tolerance.
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
    #= SOLUTION: columns outside, rows inside:  for j in axes(M, 2) … for i in axes(M, 1) =#
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)             # 1st index innermost
            s += M[i, j]
        end
    end
    #= END =#
    return s
end

#%% Rows vs columns
@assert sum_by_rows(A) ≈ sum_by_columns(A)
print("along rows   : "); @btime sum_by_rows($A)
print("down columns : "); @btime sum_by_columns($A)

# Explain the measured ratio using the actual array layout. Try a smaller matrix; a small gap does not imply that order never matters in cache.
