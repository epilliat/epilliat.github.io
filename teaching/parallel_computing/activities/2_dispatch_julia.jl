# Multiple dispatch — and why it is only "free" with concrete types.  ~6 minutes.
# Fill the two methods, run each #%% cell, read the numbers.
# Needs BenchmarkTools:  using Pkg; Pkg.add("BenchmarkTools")
#
# (This is a LANGUAGE activity — no Python twin: Python always looks methods up at run
#  time, so the concrete-vs-abstract contrast below has no Python equivalent. It IS the
#  point of the module.)

#%% setup
using BenchmarkTools
abstract type Shape end
struct Circle    <: Shape; r::Float64;             end
struct Rectangle <: Shape; w::Float64; h::Float64; end

#%% Your turn: one `area` method per shape — same function name, different argument types
#= SOLUTION: define area(c::Circle) = π*c.r^2  and  area(r::Rectangle) = r.w*r.h =#
area(c::Circle)    = π * c.r^2
area(r::Rectangle) = r.w * r.h
#= END =#

#%% Run me — dispatch picks the right method for each element, no if/isa
area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

#%% Run me — the catch: does the CONTAINER's type change the speed?
total(v) = sum(area, v)
radii    = rand(1000)
concrete = [Circle(r) for r in radii]         # Vector{Circle}  — one concrete type
mixed    = Shape[Circle(r) for r in radii]    # Vector{Shape}   — abstract, SAME data
@assert total(concrete) ≈ total(mixed)
print("Vector{Circle} (concrete) : "); @btime total($concrete)
print("Vector{Shape}  (abstract) : "); @btime total($mixed)
# Same data, same answer — but the abstract container is ~10× slower. With a concrete
# element type Julia knows at compile time WHICH `area` to call, and inlines it. With a
# Vector{Shape} it must look the method up AT RUN TIME, per element — exactly like Python.
# Multiple dispatch is fast only when the type is concrete. That's why type stability
# (the compilation module) is the thing that actually buys the speed.
