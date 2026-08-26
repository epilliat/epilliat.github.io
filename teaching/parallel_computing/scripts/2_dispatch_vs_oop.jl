# =============================================================================
# Multiple dispatch vs OOP — where the verb lives: left vs right of the data
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC with `pluto/2_dispatch_vs_oop.jl` — same lesson, two formats.
# Cells are delimited by `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools.
#
# The compilation module showed WHY specializing on types is fast. This module is the
# LANGUAGE MECHANISM that makes it possible — and the contrast with Python's OOP.
# Not parallelism yet; but this is what later moves a computation onto the GPU
# without changing a line.
# =============================================================================

#%% Where does the verb go?
#   OOP (Python)      shape.area()   → verb on the RIGHT; the method is INSIDE the class
#   dispatch (Julia)  area(shape)    → verb on the LEFT; a generic function, OUTSIDE the types
#
# The receiver left of the dot picks the OOP method — ONE object has a say.
# The Julia method is picked from the types of ALL the arguments.

#%% 1. OOP: the verb hangs on the right
#   class Circle(Shape):
#       def __init__(self, r): self.r = r
#       def area(self): return 3.14159 * self.r**2
#
#   Circle(2.0).area()          # dispatch on the object, left of the dot
#
# The method is locked inside the class: to add `perimeter`, you EDIT EVERY class.
#
# 🐍 "Python can do that too" — true, and worth saying out loud:
#   (a) monkey-patching (`Circle.perimeter = lambda self: ...`) mutates a shared class
#       at run time and fails on builtins / __slots__ / C types. An escape hatch.
#   (b) functools.singledispatch is the right idea — but note the name: SINGLE. It
#       chooses on the FIRST argument only. Section 3 is where that stops being enough.
# The real question is not "can it?" but "what is the DEFAULT, and what may the
# compiler ASSUME?" — section 5 answers that with a benchmark.

#%% 2. Julia: separate the data (types) from the operations (functions)
abstract type Shape end

struct Circle <: Shape
    r::Float64
end

struct Rectangle <: Shape
    L::Float64
    w::Float64
end

#%% One generic function `area`, one method per type
area(c::Circle)    = π * c.r^2
area(r::Rectangle) = r.L * r.w

#%% The dot broadcasts `area`; the right method is picked per element
area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

#%% ⚠ But look at the CONTAINER that demo built — it is the counter-example
shapes = [Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)]
typeof(shapes)                      # → Vector{Shape}: an ABSTRACT element type

#%% Concrete or not?
isconcretetype(eltype(shapes))      # → false
# Mixing types forces the common supertype. So for THIS array the method cannot be
# chosen at compile time — Julia looks it up per element, at run time, like Python.

#%% Measure it — same data, same answer, only the container's type differs
using BenchmarkTools
using InteractiveUtils   # @code_typed — auto-loaded in the REPL, not in a script
total(v) = sum(area, v)
radii    = rand(1000)
concrete = [Circle(r) for r in radii]        # Vector{Circle} → concrete
mixed    = Shape[Circle(r) for r in radii]   # Vector{Shape}  → abstract, SAME data
@assert total(mixed) ≈ total(concrete)
print("Vector{Shape}  (abstract) : "); @btime total($mixed)
print("Vector{Circle} (concrete) : "); @btime total($concrete)
# ~1 µs vs ~100 ns: an order of magnitude, zero allocations on both sides. The only
# difference is whether the compiler could know the type in advance.

#%% Why — ask the compiler what it produced
@code_typed total(mixed)      # a DYNAMIC call to `Main.area` survives into the code

#%% ...and the concrete version
@code_typed total(concrete)   # area is resolved and inlined — the lookup is GONE
# Dispatch buys speed only when the types are CONCRETE. An abstract container gives
# you the flexibility AND the Python price tag. A Vector{Shape} is not a bug — just
# know what it costs, and prefer a concrete container in a hot loop.

#%% 3. The difference that counts: ALL the arguments, not just one
# 🐍 What you write in Python, where singledispatch does NOT save you:
#
#   class Circle(Shape):
#       def collide(self, other):
#           if isinstance(other, Circle):      return "two circles"
#           elif isinstance(other, Rectangle): return "circle and rectangle"
#           else: raise TypeError(...)         # and the SAME ladder in every class
#
# The isinstance ladder is the tell: the language dispatched on one type, so YOU
# hand-write the dispatch on the other. N shapes = N classes × N branches, and a new
# type means editing them all. (The Visitor pattern exists only to fake this.)
collide(a::Circle,    b::Circle)    = "two circles"
collide(a::Circle,    b::Rectangle) = "circle and rectangle"
collide(a::Rectangle, b::Rectangle) = "two rectangles"

[collide(Circle(1.0), Circle(2.0)),
 collide(Circle(1.0), Rectangle(2.0, 3.0)),
 collide(Rectangle(1.0, 1.0), Rectangle(2.0, 2.0))]
# Each line is a method chosen from BOTH types. Adding a type adds methods without
# touching the ones above.

#%% ...but symmetry is NOT free — the pair we never defined
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    println("MethodError, as promised:\n  ", sprint(showerror, e)[1:min(end, 120)], " …")
end
# Nothing says a collision is commutative, so Julia refuses rather than assume. Say it
# explicitly, once:
collide(a::Rectangle, b::Circle) = collide(b, a)
collide(Rectangle(1.0, 1.0), Circle(2.0))
# Dispatch does exactly what you WROTE — which is why `methods(collide)` is an honest,
# complete list of what is supported.

#%% 4. Extending without editing — the "expression problem"
# A new OPERATION on existing types, without touching their definitions:
perimeter(c::Circle)    = 2π * c.r
perimeter(r::Rectangle) = 2 * (r.L + r.w)

perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])

#%% ...and a new TYPE, without touching existing code
#   struct Triangle <: Shape; b::Float64; h::Float64; end
#   area(t::Triangle) = t.b * t.h / 2
#
#              add a TYPE            add an OPERATION
#   OOP        easy (new subclass)   EDIT EVERY class
#   dispatch   easy (new struct)     easy (new function)
#
# ⚠ Don't over-claim. "Julia solves the expression problem" is too strong: Wadler's
#   version also demands compile-time exhaustiveness, and Julia gives none — a missing
#   method is a RUN-TIME MethodError. Add method ambiguities and type piracy. Dispatch
#   wins the EXTENSIBILITY half, convincingly; it is not a static guarantee.

#%% 5. Why this matters for speed
# When you call area(Circle(2.0)) Julia knows the argument type, picks the method, and
# COMPILES a specialized native version — the same mechanism as f(3.0) vs f(3).
#
#   Python obj.method()  → looked up at RUN time, every call, unspecialized
#   Julia  area(x)       → chosen at COMPILE time *IF THE TYPE IS INFERABLE*, then
#                          inlined; the lookup disappears
#
# ⚠ When it is NOT inferable (abstract container, type instability) Julia falls back to
#   run-time dispatch, exactly like Python — that is section 2's Vector{Shape}.
#   Dispatch is the mechanism; INFERENCE is the precondition.
methods(area)          # every method Julia knows for this generic function

#%% 6. Summary
# Where the method lives  Python: inside the object · Julia: outside, generic function
# Chosen from             Python: ONE type (singledispatch too) · Julia: ALL arguments
# Add a type              easy / easy
# Add an operation        Python: edit every class · Julia: add a function
# Two-argument dispatch   Python: isinstance ladder or Visitor · Julia: another method
# Link to speed           the ENGINE of specialization — but only on inferable types
#
# 🐍 python/2 runs this same Circle/Rectangle/collide example with `singledispatch` and the
#    third-party `multipledispatch` — same feature on the surface, but the lookup happens at
#    RUN time, so it buys none of the speed. It also covers why duck typing is a separate
#    question from single vs multiple dispatch.
#
# TAKEAWAY: dispatch is not just style, it is what lets Julia generate specialized
# native code, and what it needs from you is CONCRETE TYPES. We meet it again when the
# same `*` picks the GPU version by itself, just because its arguments are CuArray.
#
# WHAT'S NEXT: parallelism on the CPU — threads, and the parallel sum that lies.
