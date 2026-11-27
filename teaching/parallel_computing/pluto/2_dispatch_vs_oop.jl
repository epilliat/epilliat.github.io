### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ d2000000-0000-4a00-8000-000000000001
md"""
# Dispatch, types, and performance
### How Julia chooses a method, and when the compiler can resolve the call

Follow the 20-minute core before the optional examples. Two questions guide it:
which argument types select a method, and what information can the compiler infer inside a loop?
Python examples provide a familiar comparison; dispatch and specialization are related but distinct.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001d
begin
    using BenchmarkTools   # @btime / @benchmark — sections 3 and 4 measure what dispatch and types cost
    md"Packages loaded ✓"
end

# ╔═╡ d2000000-0000-4a00-8000-000000000003
md"""
## 1. One function, several methods

A function can have several methods. Argument types determine which method is called.
Python's ordinary class-method syntax is `shape.area()`; Julia uses `area(shape)` here.
The syntax alone does not explain a performance difference.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000005
# ...the data: a small type hierarchy
begin
    abstract type Shape end

    struct Circle <: Shape
        r::Float64
    end

    struct Rectangle <: Shape
        L::Float64
        w::Float64
    end

    md"Types defined ✓"
end

# ╔═╡ d2000000-0000-4a00-8000-000000000006
# ...the operation: one generic function `area`, one method per type
begin
    area(c::Circle)    = π * c.r^2
    area(r::Rectangle) = r.L * r.w
end

# ╔═╡ 2b32054d-cdc1-449c-90c7-8aab98109770
area(Circle(2.0)), area(Rectangle(3.0, 4.0))

# ╔═╡ d2000000-0000-4a00-8000-000000000008
md"""
## 2. Dispatch uses all argument types

The ordered pair `(Circle, Rectangle)` selects a different signature from `(Rectangle, Circle)`.
Predict whether the reverse call has a matching method before running the optional example.
Ordinary Python method lookup uses the receiver; handling the other argument needs additional logic.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000009
# dispatch on TWO arguments: the method depends on the pair of types
begin
    collide(a::Circle,    b::Circle)    = "two circles"
    collide(a::Circle,    b::Rectangle) = "circle and rectangle"
    collide(a::Rectangle, b::Rectangle) = "two rectangles"

    [collide(Circle(1.0), Circle(2.0)),
     collide(Circle(1.0), Rectangle(2.0, 3.0)),
     collide(Rectangle(1.0, 1.0), Rectangle(2.0, 2.0))]
end

# ╔═╡ d2000000-0000-4a00-8000-00000000000d
md"""
## 3. Same values, different element types

Both arrays below contain the same circles. The difference is their declared element type:
`Circle` or its abstract supertype `Shape`. Predict which total is faster before timing.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000016
total(v) = sum(area, v)

# ╔═╡ d2000000-0000-4a00-8000-000000000017
begin
    circles = [Circle(r) for r in rand(1000)]
    concrete = circles
    abstract_elements = Shape[c for c in circles]
    @assert concrete == abstract_elements
    @assert total(concrete) ≈ total(abstract_elements)
    md"Same values and verified totals; different element types."
end

# ╔═╡ d2000000-0000-4a00-8000-000000000018
begin
    print("Vector{Circle}: "); @btime total($concrete)
    print("Vector{Shape} : "); @btime total($abstract_elements)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000019
md"""
Historical observations on the course laptop were about 0.1 µs for `Vector{Circle}` and
1 µs for `Vector{Shape}`, with zero allocations in both cases (warmed-up BenchmarkTools minima).
Use the timings from your run.

Are the elements stored inline in the array?
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001a
isbitstype(Circle), isbitstype(Shape)

# ╔═╡ d2000000-0000-4a00-8000-00000000001c
md"""
`Vector{Circle}` stores the radii one after another. `Vector{Shape}` stores one pointer per
element, and the loop tests the type of each element before calling the right `area`.
There is no dynamic call here: with two `Shape` subtypes, Julia replaces the call by type tests.
Abstract types remain useful for generic interfaces.
"""
# ╔═╡ 55f08765-0106-49a5-a7d7-0abe97f4a0c4
md"""
## 4. Global variable or argument?

`glob` is a non-constant global without a type annotation. Compare two functions on the same
array, predict time and allocations, then inspect which types can be inferred.
This example does not describe constants or explicitly typed globals.
"""

# ╔═╡ 924a2d90-3322-4415-aa37-a919702b77ee
glob = rand(1000)  # non-constant, untyped global

# ╔═╡ 65f2c66b-226f-4e6d-804b-9637a65a77e6
function sum_global()
    s = 0.0
    for i in eachindex(glob)
        s += glob[i]
    end
    return s
end

# ╔═╡ 9a74eb76-23ae-471f-bba9-1e554d4028f3
function sum_arg(x)
    s = 0.0
    for i in eachindex(x)
        s += x[i]
    end
    return s
end

# ╔═╡ 23601861-7f44-443f-b13a-27cd20d1ab5b
md"""
### Inspect inferred types

Use `@code_warntype` to inspect the input, accumulator and return types inside the loop.
Focus on type information, not only on the diagnostic colors.
"""

# ╔═╡ e5ba66a4-5eab-46f1-99ae-99ad4e5eba4d
@code_warntype sum_global()

# ╔═╡ f140228c-e890-4b33-a784-4c0209cde97d
md"""
Compare with the version receiving the array as an argument:
"""

# ╔═╡ f11b9f37-a293-4355-b9da-aa95bb5192a2
@code_warntype sum_arg(glob)

# ╔═╡ 1340c246-2259-4b37-afdb-787ed4ae7678
md"""
### Measure the two implementations

Check equality, then compare warmed-up minimum time and allocations on the same input.
Historical observations were about 118 µs with roughly 4000 allocations versus 0.92 µs without
allocations. Unknown types can require dynamic indexing, arithmetic and boxing inside the loop.
"""

# ╔═╡ 1c3ebba3-7520-4544-b579-de2c5f7f543b
begin
    @assert sum_global() == sum_arg(glob)
    print("global  : "); @btime sum_global()
end

# ╔═╡ 5e4e8915-06a6-48ca-824f-64f3ee2645ae
@btime sum_arg($glob)

# ╔═╡ 814dc34f-c5a2-4e51-82ba-56e3a16fc55b
md"""
## Keep types visible to the compiler

- Pass working data as function arguments.
- Prefer concrete element types in performance-critical loops.

Measure first; inspect inferred types when needed. Generic method signatures can still specialize
on their actual argument types: the issue here is information available inside the loop.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000f
md"""
## Session 1 exit ticket

“Python: 0.31 s on my laptop. Julia: 0.004 s on a server. Julia is 77× faster.”
Which three checks are needed for a fair comparison?
For your project, name one block to time and the input you will use.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000010
md"""
### Next module

Threads and concurrency: distinguish waiting from computing, then split a computation and reduce
its results. The following dispatch examples are optional.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001e
md"""
## 5. Optional — the missing argument pair

We defined `collide(::Circle, ::Rectangle)`, but not its reverse. Predict the result of the next call.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001f
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    sprint(showerror, e)      # → MethodError: no method matching collide(::Rectangle, ::Circle)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000020
md"""
### Declare the reverse pair

The error shows a missing method for the ordered pair, not an assumption of symmetry.
The definition below is disabled initially. Enable it in Pluto's cell menu to add the reverse
method; the earlier cell then re-runs because Pluto is reactive.
"""

# ╔═╡ e663a7e4-bb42-4b8f-a61c-c121ba409b20
# ╠═╡ disabled = true
#=╠═╡
begin
    collide(a::Rectangle, b::Circle) = collide(b, a)      # symmetry, declared once
    collide(Rectangle(1.0, 1.0), Circle(2.0))
end
  ╠═╡ =#

# ╔═╡ d2000000-0000-4a00-8000-00000000000a
md"""
### A new operation on existing types

Add `perimeter` methods without changing the type definitions. In Python, external functions
and `singledispatch` are also possible; the Python notebook gives an example.
With class-defined methods, the operation is added to each relevant class.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000b
# a NEW operation on the existing types — their definitions stay untouched
begin
    perimeter(c::Circle)    = 2π * c.r
    perimeter(r::Rectangle) = 2 * (r.L + r.w)

    perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])
end

# ╔═╡ 33dc6971-3ba9-42e4-871b-491010f9d5d7
md"""
### List a function's methods

`methods(area)` lists the available method signatures. Which argument types does each accept?
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000e
methods(area)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─d2000000-0000-4a00-8000-000000000001
# ╠═d2000000-0000-4a00-8000-00000000001d
# ╟─d2000000-0000-4a00-8000-000000000003
# ╠═d2000000-0000-4a00-8000-000000000005
# ╠═d2000000-0000-4a00-8000-000000000006
# ╠═2b32054d-cdc1-449c-90c7-8aab98109770
# ╟─d2000000-0000-4a00-8000-000000000008
# ╠═d2000000-0000-4a00-8000-000000000009
# ╟─d2000000-0000-4a00-8000-00000000000d
# ╠═d2000000-0000-4a00-8000-000000000016
# ╠═d2000000-0000-4a00-8000-000000000017
# ╠═d2000000-0000-4a00-8000-000000000018
# ╟─d2000000-0000-4a00-8000-000000000019
# ╠═d2000000-0000-4a00-8000-00000000001a
# ╠═d2000000-0000-4a00-8000-00000000001b
# ╟─d2000000-0000-4a00-8000-00000000001c
# ╟─55f08765-0106-49a5-a7d7-0abe97f4a0c4
# ╠═924a2d90-3322-4415-aa37-a919702b77ee
# ╠═65f2c66b-226f-4e6d-804b-9637a65a77e6
# ╠═9a74eb76-23ae-471f-bba9-1e554d4028f3
# ╟─23601861-7f44-443f-b13a-27cd20d1ab5b
# ╠═e5ba66a4-5eab-46f1-99ae-99ad4e5eba4d
# ╟─f140228c-e890-4b33-a784-4c0209cde97d
# ╠═f11b9f37-a293-4355-b9da-aa95bb5192a2
# ╟─1340c246-2259-4b37-afdb-787ed4ae7678
# ╠═1c3ebba3-7520-4544-b579-de2c5f7f543b
# ╠═5e4e8915-06a6-48ca-824f-64f3ee2645ae
# ╟─814dc34f-c5a2-4e51-82ba-56e3a16fc55b
# ╟─d2000000-0000-4a00-8000-00000000000f
# ╟─d2000000-0000-4a00-8000-000000000010
# ╟─d2000000-0000-4a00-8000-00000000001e
# ╠═d2000000-0000-4a00-8000-00000000001f
# ╟─d2000000-0000-4a00-8000-000000000020
# ╠═e663a7e4-bb42-4b8f-a61c-c121ba409b20
# ╟─d2000000-0000-4a00-8000-00000000000a
# ╠═d2000000-0000-4a00-8000-00000000000b
# ╟─33dc6971-3ba9-42e4-871b-491010f9d5d7
# ╠═d2000000-0000-4a00-8000-00000000000e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
