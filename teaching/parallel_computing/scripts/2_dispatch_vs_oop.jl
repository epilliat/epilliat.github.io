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

area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

shapes = [Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)]
typeof(shapes)                      # → Vector{Shape}: an ABSTRACT element type

#%% Concrete or not?
isconcretetype(eltype(shapes))

using BenchmarkTools
using InteractiveUtils   # @code_typed — auto-loaded in the REPL, not in a script
total(v) = sum(area, v)
radii    = rand(1000)
concrete = [Circle(r) for r in radii]        # Vector{Circle} → concrete
mixed    = Shape[Circle(r) for r in radii]   # Vector{Shape}  → abstract, SAME data
@assert total(mixed) ≈ total(concrete)
print("Vector{Shape}  (abstract) : "); @btime total($mixed)
print("Vector{Circle} (concrete) : "); @btime total($concrete)


@code_typed total(mixed)      # a DYNAMIC call to `Main.area` survives into the code

#%% ...and the concrete version
@code_typed total(concrete)   # area is resolved and inlined — the lookup is GONE

collide(a::Circle,    b::Circle)    = "two circles"
collide(a::Circle,    b::Rectangle) = "circle and rectangle"
collide(a::Rectangle, b::Rectangle) = "two rectangles"

[collide(Circle(1.0), Circle(2.0)),
 collide(Circle(1.0), Rectangle(2.0, 3.0)),
 collide(Rectangle(1.0, 1.0), Rectangle(2.0, 2.0))]


try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    println("MethodError, as promised:\n  ", sprint(showerror, e)[1:min(end, 120)], " …")
end

collide(a::Rectangle, b::Circle) = collide(b, a)
collide(Rectangle(1.0, 1.0), Circle(2.0))

perimeter(c::Circle)    = 2π * c.r
perimeter(r::Rectangle) = 2 * (r.L + r.w)

perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])

methods(area)       
