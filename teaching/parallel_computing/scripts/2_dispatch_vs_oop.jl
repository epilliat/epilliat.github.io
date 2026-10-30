# Multiple dispatch vs OOP — where the verb lives (a language point, no parallelism yet). ENSAI 3A.
# ⚠ KEEP IN SYNC with `pluto/2_dispatch_vs_oop.jl`. Cells: `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools.
# For each cell: predict the output, then run it.
using BenchmarkTools
using InteractiveUtils   # @code_typed/_warntype (auto-loaded in the REPL only)

#%% 1. One argument: one function, one method per type
abstract type Shape end

struct Circle <: Shape
    r::Float64
end

struct Rectangle <: Shape
    L::Float64
    w::Float64
end

area(c::Circle)    = π * c.r^2
area(r::Rectangle) = r.L * r.w

c = Circle(0.5)
area(c)

shapes = [Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)]
area.(shapes)

#%% 2. Two arguments: dispatch on the PAIR of types
collide(a::Circle,    b::Circle)    = "two circles"
collide(a::Circle,    b::Rectangle) = "circle and rectangle"
collide(a::Rectangle, b::Rectangle) = "two rectangles"

[collide(Circle(1.0), Circle(2.0)),
 collide(Circle(1.0), Rectangle(2.0, 3.0)),
 collide(Rectangle(1.0, 1.0), Rectangle(2.0, 2.0))]

#%% The pair we did not define: what happens?
# (run it BEFORE the next cell: once the method exists, the error is gone)
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    showerror(stdout, e)
end

#%% Add the missing pair in one line
collide(a::Rectangle, b::Circle) = collide(b, a)
collide(Rectangle(1.0, 1.0), Circle(2.0))

#%% A new operation, without touching the types
perimeter(c::Circle)    = 2π * c.r
perimeter(r::Rectangle) = 2 * (r.L + r.w)

perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])

#%% Who knows which methods? How many for `area`?
methods(area)

#%% 3. Dispatch and speed: what container did `shapes` build?
typeof(shapes)

#%% Is its element type concrete?
isconcretetype(eltype(shapes))

#%% Same data, two containers: predict which is faster, and by how much
total(v) = sum(area, v)
radii    = rand(1000)
concrete = [Circle(r) for r in radii]
mixed    = Shape[Circle(r) for r in radii]
@assert total(mixed) ≈ total(concrete)
print("Vector{Shape}  : "); @btime total($mixed)
print("Vector{Circle} : "); @btime total($concrete)

#%% Why? Find the call to `area` in each
@code_typed total(mixed)

#%% …and here?
@code_typed total(concrete)

#%% 4. Type instability — same sum, global vs argument
glob = rand(1000)

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

#%% Spot the red `Any`
@code_warntype sum_global()

#%% …and here?
@code_warntype sum_arg(glob)

#%% Predict the gap, then measure
@benchmark sum_global()

#%% The argument version
@benchmark sum_arg($glob)
