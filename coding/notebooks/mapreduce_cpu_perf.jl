### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 6a11ce45-a560-4bac-a1da-deacc3bb9084
begin
	using BenchmarkTools
end

# ╔═╡ 4090abca-7991-4a23-8355-07eaef7c006b
md"""
# Performance Analysis of mapreduce
"""

# ╔═╡ 5c4354c4-054d-44fa-a20c-6ddcbb159d32
md"""
In this notebook, we study Julia Base's `sum` and `mapreduce` performance. Our optimized implementation achieves:
- 20% speedup over Base
- Support for general types
- Consistent performance across all sizes (small to large arrays)
- Preserved floating-point precision via tree reduction

We develop the solution incrementally, starting from naïve versions and progressively troubleshooting performance issues.
"""

# ╔═╡ 39d4d1ce-687f-40c1-8706-04173f64e949
md"""
## Naïve sum vs Base.sum

Let's try a naïve version of sum.
"""

# ╔═╡ 2be9d620-d513-11f0-242d-59b69cad5eb7
function mysum(a::AbstractArray{T}) where T
	n = length(a)
	s = T(0)
	for i in (1:n)
		s+=a[i]
	end
	return s
end

# ╔═╡ 09ba4f2c-1fdf-4d79-bafa-a58ca75ab6b0
begin
	n = 2^20
	a = rand(Float32,n)
end

# ╔═╡ c3c1f273-6f8f-4ed8-90d8-0d6d69ebf0e1
@btime sum($a); @btime mysum($a)

# ╔═╡ 12007cd7-b902-4631-8b8d-ae507101c4a4
md"""
## Too bad.

This is 5 times slower than Base !
Let's try with `@simd` which performs aggressive optimisations by reordering operations
"""

# ╔═╡ c3ff15b5-c331-40eb-a96c-a1577c7d61b9
function mysum_simd(a::AbstractArray{T}) where T
	n = length(a)
	s = T(0)
	@simd for i in (1:n)
		s+=a[i]
	end
	return s
end

# ╔═╡ 60409f86-75ae-463c-8993-5a3ef42acfc0
@btime sum($a);  @btime mysum_simd($a)

# ╔═╡ fa864919-4e41-4493-8a41-9fd489d7a38f
md"""
## This is much better !?

Wait... this is much better than default sum ? Something must be wrong somewhere, let's **check the precision**:
"""

# ╔═╡ 15af4285-123e-4aad-ac06-bb0a3fdf304c
begin
	eps1=(abs(mysum_simd(a) - sum(Float64.(a))))
	eps2 =(abs(sum(a) - sum(Float64.(a))))
	print("error for Base.sum: $eps2,\n error for new version: $eps1")
end

# ╔═╡ 2dc9be65-df67-4738-a5d2-2a465e1e7464
md"""
### We loose in precision !

Wow, **1000x less precise**. Let's try smaller arrays of length 1024 !
"""

# ╔═╡ 3e05eeb8-2369-4a6d-b27d-298ba0f072c9
@btime mysum_simd($(a[1:2^10])); @btime sum($(a[1:2^10]))

# ╔═╡ 04563ba1-6b6c-47e6-8bc4-2a71c32bf64e
md"""
### What's happening ? 
 
Here we have no problem of precision and **we are much faster**. Let's try the sum of Ints for large arrays where the result does not suffer from numerical errors.
"""

# ╔═╡ d406ea54-97b4-492d-9166-3e445830df2c
begin 
	b = rand(Int, n)
	@btime mysum_simd($(b)); @btime sum($(b))
	print("is equal to Base.sum: ", mysum_simd(b) == sum(b))
end

# ╔═╡ bc941554-5aa6-4692-8958-47da1f2fd67b
md"""
## Alignment problems

Ok there is a serious optimization in Julia Base. We think this is because of simd alignment issues.
Note that something like this is really subperforming
"""

# ╔═╡ fd2fea00-7f6d-46c3-937d-4029246daa81
@btime mysum_simd($(a[1:31])); @btime sum($(a[1:31]))

# ╔═╡ fb1cdba5-0285-4a61-8d06-748178e00825
md"""
Ok so simd is really good on loops that maker power of 2 iterations.
"""

# ╔═╡ 983c4ee3-4306-4eca-990a-f5e24e7c6bd0
md"""
## Smaller arrays

For very small arrays that are of size under power of 2 like 17, 31, it might be worth loading everything in the first place and then doing the reduction on registers. This is maybe a simd limitation ?
"""

# ╔═╡ beabc81f-a358-4c91-9cdc-78792941c7bb
@inline function vload(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    ptr = reinterpret(Ptr{NTuple{Nitem,T}}, pointer(A) + (idx - 1) * sizeof(T))
    return unsafe_load(ptr, 1)
end

# ╔═╡ 868b1db5-5add-478d-99a4-f78996ce6067
@btime +(vload($a, 1, Val(31))...)

# ╔═╡ a282afd2-47bd-4d53-937b-cc66da6e25af
md"""
This is very fast !
"""

# ╔═╡ 06eaf47e-54f6-455a-a0af-36ed6e072e33
md"""
## Toward fast mapreduce

Let's write a general mapreduce function where we reduce over blocks of length Nblocks. we aggregate the sums in a tree fashion where the number of branch at each level is Nbranch.

For the sum of each block, we put the possibility of giving a neutral as a Val. This can be much faster with simd for very small arrays as shown above. When neutral is not given, we just mapreduce over (1:32) to get a first value and stay well aligned

In Julia Base:

- Nblocks = 1024
- Nbranch = 2

This is generic, but commutative operations (non-Floating point ops, or even multiplication for Floats) we can take Nbranch = 2 (or anything else) and Nblocks = 2^100 (largest possible).
"""

# ╔═╡ 0e06ccd9-02ed-4a8f-9bc2-4737096e088d
md"""
## Small Arrays

Let's start with small arrays. For Float32, we say its small if the length is smaller than 32. In that case, we can specialize 32 optimized functions with Val.

**The idea** is simple: load everything into a tuple, then extract everything into registers, finally sum everything.
"""

# ╔═╡ 83eb3442-37fa-4fd9-b6f2-0fecff4a216c
@generated function mapreduce_fixedsize(f, op, A, L, ::Val{N}) where {N}
    quote
        v = vload(A, L, Val($N))
        Base.Cartesian.@nexprs $N i -> v_i = v[i]
        s = f(v_1)
        Base.Cartesian.@nexprs $(N-1) i -> s = op(s, f(v_{i+1}))
        return s
    end
end

# ╔═╡ 1009a583-dc24-46c2-b8d4-5b454c0eab69
@btime mapreduce_fixedsize(identity, +, $a, 1, Val(32)); @btime sum($(a[1:32]))

# ╔═╡ a426d698-fc06-4bba-a428-95800179fbeb
md"""
### The problem

At run time, evaluating a Val is costly. We completely loose the performance.
"""

# ╔═╡ 15261ecd-6752-45a6-a3f0-e356ed12407a
function sum_fixedsize(A, L, N)
	return mapreduce_fixedsize(identity, +, A,1, Val(N))
end

# ╔═╡ 4969edbb-35a9-405c-a58a-a3fa00338fc9
@btime sum_fixedsize(a,1,32)

# ╔═╡ 19831350-ce93-4200-81c4-6880c8e2ee24
md"""
### The solution

We want to apply `sum_smallarray` with a known `Val`. For that, a solution is to build the expression:
```julia
if R-L+1 <= 16
    if R-L+1 <= 4
        ... for example: mapreduce(f, op, A, L, Val(3)) where 3 is fixed!
    else
    end
else
end
```

This is costly to write manually, so we use a generated function.
"""

# ╔═╡ 26653f8f-940f-4666-b1b8-23e470224d0b
macro mapreduce_boundedsize(f, op, A, L, N, Nmax)
    function build_if_tree(start, stop)
        if stop - start + 1 == 1
            # Cas de base : une seule valeur
            n = start
            return :(mapreduce_fixedsize($f, $op, $A, $L, Val($n)))
        else
            # Divise en deux
            mid = start + (stop - start + 1) ÷ 2 - 1
            left = build_if_tree(start, mid)
            right = build_if_tree(mid + 1, stop)
            
            return quote
                if $N <= $mid
                    $left
                else
                    $right
                end
            end
        end
    end
    
    tree = build_if_tree(1, Nmax)
    
    return esc(tree)
end

# ╔═╡ 3e957a97-52c1-4592-9438-3939d0bd5a49
@macroexpand @mapreduce_boundedsize(identity, +, a, 1, 32, 32)

# ╔═╡ 7f5b76cb-7452-471a-959d-a21ca7cf89b2
@generated function mapreduce_boundedsize(f, op, A, L, R, ::Val{Nmax}) where Nmax
	quote
		N = R-L+1
		@mapreduce_boundedsize(f, op, A,L,N, $Nmax)
	end
end

# ╔═╡ 461325ac-c428-400f-8fe6-3e77aad30ff7
begin
	for R in 1:32
		println("mapreduce_fixedsize, mapreduce_boundedsize, mysum_simd over 1:$R")
		@btime mapreduce_fixedsize(identity, +, $a, 1, $(Val(R)))
		@btime mapreduce_boundedsize(identity, +, $a, 1, $R,Val(32))
		@btime mysum_simd($(a[1:R]))
		println("")
	end
end

# ╔═╡ 82f79035-cdd0-416f-b57b-fb4d3fce9aa6
md"""
Fixed size 
"""

# ╔═╡ 3e462fa3-dca1-4013-9958-beac3c2b9751
md"""
### Conclusion

for small arrays, there is a slight overhead for getting the size before calling a static sized function, but we still get better performances than simd with neutral element, except for 32.

A few comments:

- The optimal size 32 may vary in function of type size. We would expect much larger size for UInt8 since a huge part of performance gain is due to loading items in contiguous memory places

Ideas of algo comming next:

- If we have a neutral and array size >= 32: use simd only
- If we have a neutral (or not) and array size < 32: just do the loading above
- If we do not have a neutral and array size >= 32: load the data as above for the 32 first elements to get a first value, and start at aligned position 33.
We should be careful that 33 is also well aligned for other types. Otherwise we should take another value.
"""

# ╔═╡ fd0b8c62-d6b2-4d23-ae30-bdb16d98c8e6
md"""
## Medium Sized Arrays
"""

# ╔═╡ df5bd6be-1297-4bf1-9bd5-9c08b814c33f
@noinline function mapreduce_block(f::F, op, A::AbstractArray{T}, L, R, init=nothing) where {T, F<: Function}
    if R-L+1<32
        return mapreduce_boundedsize(f,op,A,L,R,Val(32))
	end
	if isnothing(init)
		s = mapreduce_fixedsize(f,op,A,L,Val(32))
		start = L+32
	else
		s= init
		start=L
	end
    @simd for i in start:R
        @inbounds s = op(s, f(A[i]))
    end
    return s
end

# ╔═╡ 788a83a3-2d98-40d4-9961-d49578f4246e
md"""
## Alignment and neutral impact on performance

Putting a neutral leads to better performances. Moreover, the alignment has a significant role in the final performance. 

Below:

- First value: when a neutral is precised. 
- Second: when no neutral is precised. When no neutral is precised, we need a block of length 1025 !
"""

# ╔═╡ 34706d90-169a-4a77-bc96-2ce9294f2b6d
@btime mapreduce_block(identity, +, $a, 1,32, 0.0f0)

# ╔═╡ bd28225d-0363-403e-b910-46d61b5c9cde
@btime mapreduce_block(identity, +, $a, 1,1024, 0.0f0); @btime mapreduce_block(identity, +, $a, 1,1024)

# ╔═╡ 4f29066d-cb16-4927-b8b5-7485d3d7a967
@btime mapreduce_block(identity, +, $a, 2,1024, 0.0f0); @btime mapreduce_block(identity, +, $a, 2,1024)

# ╔═╡ c95f8dd5-843b-4206-9fb6-b3799619e37d
@btime mapreduce_block(identity, +, $a, 1,1023, 0.0f0); @btime mapreduce_block(identity, +, $a, 1,1023)

# ╔═╡ 88f322b1-c776-4848-9fef-ae33290275be
@btime mapreduce_block(identity, +, $a, 33,$(1024+32), 0f0); @btime mapreduce_block(identity, +, $a, 33,$(1025+32))

# ╔═╡ 41df341a-06e0-46a7-a076-cc12924c0a8c
@btime mapreduce_block(identity, +, $a, 1, $n, 0.0f0)

# ╔═╡ 1b159ac8-cfac-4429-8086-ce7a950a665a
function mapreduce_tree(f::F,op::O, A::AbstractArray{T}, L, R, ::Val{Nblock}, ::Val{Nbranch}, neutral=nothing) where {T,Nblock,Nbranch, F<:Function, O<:Function}
    R - L + 1 <= Nblock && return mapreduce_block(f,op, A, L, R, neutral)

    i = L
    s1 = mapreduce_block(f,op, A, L, L + Nblock - 1, neutral)
    i += Nblock
    u = (((R - L + 1 - Nblock) >> trailing_zeros(Nbranch * Nblock)) << trailing_zeros(Nblock))

    if u < Nblock
        while i <= R - Nblock + 1
            s2 = mapreduce_block(f,op, A, i, i + Nblock - 1, neutral)
            s1 = op(s1, s2)
            i += Nblock
        end
    else
        while i <= R - u + 1
            s2 = mapreduce_tree(f,op, A, i, i + u - 1, Val(Nblock), Val(Nbranch), neutral)
            s1 = op(s1, s2)
            i += u
        end
    end
    i <= R && (s1 = op(s1, mapreduce_block(f,op, A, i, R, neutral)))
    return s1
end

# ╔═╡ 65262ab5-52e8-40df-b491-ae4cd9667252
md"""
## Final Benchmarks

we evaluate mapreduce_tree according two criteria:

- Its precision on Floating points
- Its speed with @btime

To tune the two parameters Nblock and Nbranch, we have the following compromise:

- We want Nblock as large as possible (e.g. 2^63) to be as fast as possible
- If Nblock is small, we want Nbranch to be as large as possible to get a tree as small as possible
- But we still dont want it to be too large for Floating points. Taking e.g. Nblocks=8192 and Nbranch=16 seems to be a good compromise.
- For operations on Ints, or *, min, max,... we could take Nblocks=2^60, or directly call mapreduce_block.
"""

# ╔═╡ 24dff7d0-2904-4014-b264-67d5383c914b
md"""
### Precision

We get a much better precision on Float32
"""

# ╔═╡ 296ae07b-e8bb-42a2-9778-e2324363c6e7
print("mapreduce_tree: $(mapreduce_tree(identity,+, a, 1, n, Val(8192), Val(16), 0.0f0)- sum(Float64.(a))) \n
Base.mapreduce: $(sum(a)- sum(Float64.(a)))")

# ╔═╡ e0f61d0e-d75e-4031-a862-8ccbeb943d65
md"""
## Time speed
"""

# ╔═╡ 96f47ab2-5edb-4622-92ce-9192f5de4cc6
begin
@btime mapreduce_tree(identity,+, $a, 1, $n, Val(8192), Val(16), 0.0f0)
#@btime sum($a)
end

# ╔═╡ fe362682-5554-4ace-bfd2-1d45effcc9f5
@btime sum($a)

# ╔═╡ 77bd328d-b984-418a-9775-05711c3c5d7c
begin
@btime mapreduce_tree(identity,+, $a, 1, $n, Val(2^30), Val(16), 0.0f0)
#@btime sum($a)
end

# ╔═╡ 84991c3f-2657-4f30-8e99-1e3cc88c94ba
begin
@btime mapreduce_tree(identity,+, $a, 1, 12345, Val(2^30), Val(2), 0.0f0)
#@btime sum($a)
end

# ╔═╡ 9e4c5bf5-b681-4f72-855d-de1db2b4ce6d
@btime sum($(a[1:12345]))

# ╔═╡ e09b711e-028c-4a59-bacd-b684b4800df2
md"""
## Conclusion

We increase the speed of sum and general reductions by something of order 20\% on all types of arrays.
We should be careful to optimize pairwise reduction for some operators. For example, we found out that it's much better to do 
```julia
ai = A[i]
if ai< s
	s=ai
end
```

then simply 
```julia
s=min(s,A[i])
```

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[compat]
BenchmarkTools = "~1.6.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "10fbcacea77e5f06204aa57f49837bbf31a9edbf"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "5b6bb73f555bc753a6153deec3717b8904f5551c"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.3.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "79529b493a44927dd5b13dde1c7ce957c2d049e4"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.0"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─4090abca-7991-4a23-8355-07eaef7c006b
# ╟─5c4354c4-054d-44fa-a20c-6ddcbb159d32
# ╟─6a11ce45-a560-4bac-a1da-deacc3bb9084
# ╟─39d4d1ce-687f-40c1-8706-04173f64e949
# ╠═2be9d620-d513-11f0-242d-59b69cad5eb7
# ╠═09ba4f2c-1fdf-4d79-bafa-a58ca75ab6b0
# ╠═c3c1f273-6f8f-4ed8-90d8-0d6d69ebf0e1
# ╟─12007cd7-b902-4631-8b8d-ae507101c4a4
# ╟─c3ff15b5-c331-40eb-a96c-a1577c7d61b9
# ╠═60409f86-75ae-463c-8993-5a3ef42acfc0
# ╟─fa864919-4e41-4493-8a41-9fd489d7a38f
# ╠═15af4285-123e-4aad-ac06-bb0a3fdf304c
# ╠═2dc9be65-df67-4738-a5d2-2a465e1e7464
# ╠═3e05eeb8-2369-4a6d-b27d-298ba0f072c9
# ╟─04563ba1-6b6c-47e6-8bc4-2a71c32bf64e
# ╠═d406ea54-97b4-492d-9166-3e445830df2c
# ╠═bc941554-5aa6-4692-8958-47da1f2fd67b
# ╠═fd2fea00-7f6d-46c3-937d-4029246daa81
# ╠═fb1cdba5-0285-4a61-8d06-748178e00825
# ╟─983c4ee3-4306-4eca-990a-f5e24e7c6bd0
# ╠═beabc81f-a358-4c91-9cdc-78792941c7bb
# ╠═868b1db5-5add-478d-99a4-f78996ce6067
# ╟─a282afd2-47bd-4d53-937b-cc66da6e25af
# ╟─06eaf47e-54f6-455a-a0af-36ed6e072e33
# ╟─0e06ccd9-02ed-4a8f-9bc2-4737096e088d
# ╠═83eb3442-37fa-4fd9-b6f2-0fecff4a216c
# ╠═1009a583-dc24-46c2-b8d4-5b454c0eab69
# ╟─a426d698-fc06-4bba-a428-95800179fbeb
# ╠═15261ecd-6752-45a6-a3f0-e356ed12407a
# ╠═4969edbb-35a9-405c-a58a-a3fa00338fc9
# ╟─19831350-ce93-4200-81c4-6880c8e2ee24
# ╟─26653f8f-940f-4666-b1b8-23e470224d0b
# ╠═3e957a97-52c1-4592-9438-3939d0bd5a49
# ╠═7f5b76cb-7452-471a-959d-a21ca7cf89b2
# ╠═461325ac-c428-400f-8fe6-3e77aad30ff7
# ╠═82f79035-cdd0-416f-b57b-fb4d3fce9aa6
# ╟─3e462fa3-dca1-4013-9958-beac3c2b9751
# ╟─fd0b8c62-d6b2-4d23-ae30-bdb16d98c8e6
# ╠═df5bd6be-1297-4bf1-9bd5-9c08b814c33f
# ╟─788a83a3-2d98-40d4-9961-d49578f4246e
# ╠═34706d90-169a-4a77-bc96-2ce9294f2b6d
# ╠═bd28225d-0363-403e-b910-46d61b5c9cde
# ╠═4f29066d-cb16-4927-b8b5-7485d3d7a967
# ╠═c95f8dd5-843b-4206-9fb6-b3799619e37d
# ╠═88f322b1-c776-4848-9fef-ae33290275be
# ╠═41df341a-06e0-46a7-a076-cc12924c0a8c
# ╠═1b159ac8-cfac-4429-8086-ce7a950a665a
# ╟─65262ab5-52e8-40df-b491-ae4cd9667252
# ╟─24dff7d0-2904-4014-b264-67d5383c914b
# ╠═296ae07b-e8bb-42a2-9778-e2324363c6e7
# ╟─e0f61d0e-d75e-4031-a862-8ccbeb943d65
# ╠═96f47ab2-5edb-4622-92ce-9192f5de4cc6
# ╠═fe362682-5554-4ace-bfd2-1d45effcc9f5
# ╠═77bd328d-b984-418a-9775-05711c3c5d7c
# ╠═84991c3f-2657-4f30-8e99-1e3cc88c94ba
# ╠═9e4c5bf5-b681-4f72-855d-de1db2b4ce6d
# ╟─e09b711e-028c-4a59-bacd-b684b4800df2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
