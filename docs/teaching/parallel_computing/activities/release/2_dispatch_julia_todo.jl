# Dispatch and element types (Julia, optional).  10 min.
# Fill the two methods, run each #%% cell.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")

#%% setup
using BenchmarkTools
abstract type Shape end
struct Circle <: Shape
    r::Float64
end
struct Rectangle <: Shape
    L::Float64
    w::Float64
end

#%% Your turn: one `area` method per shape
# ✏️ your turn: define one area method per concrete shape; use its fields to compute the area
error("remove this line and write your code")

#%% One call per shape
area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

#%% Concrete vs abstract container
total(v) = sum(area, v)
circles = [Circle(r) for r in rand(1000)]
concrete = circles
abstract_elements = Shape[c for c in circles]
# Predict which call is faster before running the timings. Both arrays contain only circles.
@assert concrete == abstract_elements
@assert total(concrete) ≈ total(abstract_elements)
print("Vector{Circle}: "); @btime total($concrete)
print("Vector{Shape} : "); @btime total($abstract_elements)
# Explain what the element type lets the compiler infer. A fixed slowdown is not assumed.
