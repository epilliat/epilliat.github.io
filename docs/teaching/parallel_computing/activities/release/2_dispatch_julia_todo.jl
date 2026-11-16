# Dispatch and element types (Julia, optional).  10 min.
# Fill the two methods, run each #%% cell.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
abstract type Shape end
struct Circle    <: Shape; r::Float64;             end
struct Rectangle <: Shape; L::Float64; w::Float64; end

#%% Your turn: one `area` method per shape
# ✏️ your turn: define one area method per concrete shape; use its fields to compute the area
error("remove this line and write your code")

#%% One call per shape
area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

#%% Concrete vs abstract container
total(v) = sum(area, v)
radii    = rand(1000)
concrete = [Circle(r) for r in radii]         # Vector{Circle}  — one concrete type
mixed    = Shape[Circle(r) for r in radii]    # Vector{Shape}   — abstract, SAME data
@assert total(concrete) ≈ total(mixed)
print("Vector{Circle} (concrete) : "); @btime total($concrete)
print("Vector{Shape}  (abstract) : "); @btime total($mixed)
