# Dispatch, types, and performance — method selection and compiler inference. ENSAI 3A.
# ⚠ KEEP IN SYNC with `pluto/2_dispatch_vs_oop.jl`. Cells: `#%%` (Alt+Enter in VS Code).
using BenchmarkTools
using InteractiveUtils

#%% 1. One function, several methods
abstract type Shape end
struct Circle <: Shape
    r::Float64
end
struct Rectangle <: Shape
    L::Float64
    w::Float64
end
area(c::Circle) = π * c.r^2
area(r::Rectangle) = r.L * r.w
area(Circle(2.0)), area(Rectangle(3.0, 4.0))

#%% 2. Dispatch uses all argument types
collide(a::Circle, b::Circle) = "two circles"
collide(a::Circle, b::Rectangle) = "circle and rectangle"
collide(a::Rectangle, b::Rectangle) = "two rectangles"
collide(Circle(1.0), Rectangle(2.0, 3.0))
# If the arguments are swapped, is the corresponding method defined?

#%% 3. Same values, different element types — predict before timing
circles = [Circle(r) for r in rand(1000)]
concrete = circles
abstract_elements = Shape[c for c in circles]
total(v) = sum(area, v)
@assert concrete == abstract_elements
@assert total(concrete) ≈ total(abstract_elements)

#%% Measure the two containers — warmed-up minimum time and allocations
print("Vector{Circle}: "); @btime total($concrete)
print("Vector{Shape} : "); @btime total($abstract_elements)

#%% Are the elements stored inline?
# Circle: radii stored one after another. Shape: one pointer per element,
# and a type test on each element before calling the right `area`.
isbitstype(Circle), isbitstype(Shape)

#%% 4. Global variable or argument — same input, predict time and allocations
glob = rand(1000)  # non-constant global without a type annotation
function sum_global()
    s = 0.0
    for i in eachindex(glob)
        s += glob[i]
    end
    return s
end
function sum_arg(x)
    s = 0.0
    for i in eachindex(x)
        s += x[i]
    end
    return s
end
@assert sum_global() == sum_arg(glob)

#%% Inspect inferred types — focus on input, accumulator and return
@code_warntype sum_global()
@code_warntype sum_arg(glob)

#%% Measure global and argument versions — warmed-up minimum time and allocations
print("global  : "); @btime sum_global()
print("argument: "); @btime sum_arg($glob)

#%% 5. Optional — the missing argument pair
# Run before defining the reverse pair in the next cell.
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch error
    showerror(stdout, error)
    println()
end

#%% Declare the reverse pair
collide(a::Rectangle, b::Circle) = collide(b, a)
@assert collide(Rectangle(1.0, 1.0), Circle(2.0)) == "circle and rectangle"

#%% A new operation on existing types
perimeter(c::Circle) = 2π * c.r
perimeter(r::Rectangle) = 2 * (r.L + r.w)
perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])

#%% List a function's methods
methods(area)
