# =============================================================================
# Multiple dispatch vs OOP — where the verb lives: left vs right of the data
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC: this is the plain-script twin of `pluto/2_dispatch_vs_oop.jl`.
#   The Pluto notebook and this script hold the SAME lesson in two formats — edit
#   BOTH together whenever the content changes, so they never drift apart.
# -----------------------------------------------------------------------------
# Cells are delimited by `#%%`. In VS Code (Julia extension) run a cell with
# Alt+Enter / Ctrl+Enter — the value of the last expression shows inline and the
# definitions stay in the REPL. Needs BenchmarkTools (section 2 measures).
#
# In the compilation module we saw WHY the same computation can be 100× faster:
# compiling and specializing on the TYPES removes the per-operation overhead. Here we
# look at the LANGUAGE MECHANISM that makes that specialization possible — multiple
# dispatch — contrasted head-on with the object-oriented (OOP) style you know from Python.
#
# 🧭 This is a language point, not parallelism yet. Threads come right after — but
# dispatch is what will make them elegant (and what later moves a computation onto
# the GPU without changing the code).
# =============================================================================

#%% Where does the verb go?
# The SAME idea — "the area of a shape" — written two ways:
#
#   OOP (Python)      shape.area()   → the verb is on the RIGHT, after the dot
#   dispatch (Julia)  area(shape)    → the verb is on the LEFT, prefix
#
# - OOP: the method is OWNED by the object — it lives INSIDE the class.
# - Julia: `area` is a GENERIC FUNCTION, living OUTSIDE the types.
#
# The receiver left of the dot decides the OOP method. The verb on the left decides
# the Julia method — from the types of ALL its arguments.

#%% 1. OOP: the verb hangs on the right
# Python's object-oriented style: the method belongs to the object, called as
# object.method(...). The choice is made on ONE object — left of the dot.
#
#   class Shape:
#       def area(self): ...
#
#   class Circle(Shape):
#       def __init__(self, r): self.r = r
#       def area(self): return 3.14159 * self.r**2
#
#   class Rectangle(Shape):
#       def __init__(self, L, w): self.L, self.w = L, w
#       def area(self): return self.L * self.w
#
#   Circle(2.0).area()      # dispatch on the object (left of the dot)
#
# The method is LOCKED INSIDE the class. To add an operation (say `perimeter`), you
# must EDIT EVERY class.
#
# 🐍 "NO! Python can do that too" — and you'd be right. Say it out loud now, because
# the honest answer is what makes the rest of this module land. Two counter-arguments,
# and what each really costs:
#
#   (a) MONKEY-PATCHING — bolt the method on from outside:
#           Circle.perimeter = lambda self: 2 * 3.14159 * self.r
#       It works. It also mutates a class other code shares, at run time, invisibly —
#       and it fails outright on builtins (`int`, `list`) and on anything with
#       __slots__ or a C extension type. It's an escape hatch, not a design.
#
#   (b) functools.singledispatch — a real generic function, verb on the left:
#           @singledispatch
#           def area(s): raise NotImplementedError
#           @area.register
#           def _(s: Circle): return 3.14159 * s.r**2
#       This is genuinely the right idea. Note the name: SINGLE dispatch. It chooses on
#       the type of ONE argument — the first. That limitation is not an accident of the
#       library; it's the whole point of section 3, where one argument stops being enough.
#
# So the interesting question is NOT "can Python do it?" (it can, awkwardly) but
# "what does the language make DEFAULT, and what does the compiler get to ASSUME?"
# Keep that question in mind — section 5 answers it with a benchmark.

#%% 2. Julia: the verb goes first, and sees the types
# In Julia we SEPARATE the data (types) from the operations (functions). First the
# data: a small type hierarchy.
abstract type Shape end

struct Circle <: Shape
    r::Float64
end

struct Rectangle <: Shape
    L::Float64
    w::Float64
end

#%% ...the operation: one generic function `area`, one method per type
area(c::Circle)    = π * c.r^2
area(r::Rectangle) = r.L * r.w

#%% the . broadcasts `area` over each element — the right method is picked per type
area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

#%% ⚠ Look at the type of that array before you believe the happy story
# That demo is pretty. It is also, exactly as written, the COUNTER-EXAMPLE to this
# module's thesis. Ask what container it built:
shapes = [Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)]
typeof(shapes)                      # → Vector{Shape}  ... an ABSTRACT element type!

#%% Concrete or not?
isconcretetype(eltype(shapes))      # → false. Julia CANNOT know what area() will hit.
# Mixing Circles and Rectangles forces the common supertype: Shape. So for THIS array,
# the method can NOT be chosen at compile time — Julia must look it up at run time,
# per element, exactly like Python. Our showcase demo does dynamic dispatch.

#%% So measure it — same data, same answer, one difference: the container's type
using BenchmarkTools
using InteractiveUtils   # @code_typed — auto-loaded in the REPL, NOT in a plain script
total(v) = sum(area, v)
const radii    = rand(1000)
const concrete = [Circle(r) for r in radii]        # Vector{Circle} → concrete
const mixed    = Shape[Circle(r) for r in radii]   # Vector{Shape}  → abstract, SAME data
@assert total(mixed) ≈ total(concrete)
print("Vector{Shape}  (abstract) : "); @btime total($mixed)
print("Vector{Circle} (concrete) : "); @btime total($concrete)
# Roughly 1 µs vs 100 ns → the concrete container wins by about an ORDER OF MAGNITUDE
# (measured 7-15× here depending on machine load; the concrete version is so fast that
# the ratio moves). Identical values, identical result, ZERO allocations on both sides.
# The ONLY difference is whether the compiler could know the type in advance.

#%% Why: ask the compiler what it produced
# In the abstract case the IR contains a DYNAMIC call to `Main.area` — the type isn't
# known, so the lookup survives into the running code. Compare the two:
@code_typed total(mixed)      # look for the dynamic `Main.area` call

#%% ...and the concrete version
@code_typed total(concrete)   # area is resolved, inlined — the lookup is GONE
# THIS is the module in one measurement, and it links straight back to @code_warntype:
# dispatch buys you speed only when the types are CONCRETE. Abstract containers give
# you the flexibility AND the Python price tag.
#
# 💡 Practical rule: a `Vector{Shape}` is not a bug — sometimes you truly need a mixed
# bag. But know what it costs, and prefer a concrete container (or a Union, which Julia
# can union-split) in a hot loop.

#%% 3. The difference that counts: all arguments, not just one
# Because the verb is on the left, it can look at EVERY argument — not only the first.
# A function collide(a, b) can choose a different method depending on the PAIR of
# types (a, b) — this is MULTIPLE DISPATCH. With a.collide(b), only `a` has a say;
# the second argument gets none.
#
# 🐍 FEEL THE PAIN FIRST — this is what you write in Python, and singledispatch does
# NOT save you (it dispatches on `self`/the 1st argument only, so the second type has
# to be re-discovered by hand):
#
#   class Circle(Shape):
#       def collide(self, other):
#           if isinstance(other, Circle):      return "two circles"
#           elif isinstance(other, Rectangle): return "circle and rectangle"
#           elif isinstance(other, Triangle):  return "circle and triangle"
#           else: raise TypeError(f"Circle vs {type(other)}?")
#
#   class Rectangle(Shape):
#       def collide(self, other):
#           if isinstance(other, Circle):      return "circle and rectangle"
#           elif isinstance(other, Rectangle): return "two rectangles"
#           ...                                # the SAME ladder again, per class
#
# The isinstance ladder is the tell: the language dispatched on ONE type, so YOU
# hand-write the dispatch on the other. With N shapes it's N classes × N branches,
# and adding Triangle means EDITING EVERY existing class — the very thing OOP was
# supposed to make easy. (The classic escape is the Visitor pattern: a whole design
# pattern that exists only to fake double dispatch in a single-dispatch language.)
collide(a::Circle,    b::Circle)    = "two circles"
collide(a::Circle,    b::Rectangle) = "circle and rectangle"
collide(a::Rectangle, b::Rectangle) = "two rectangles"

[collide(Circle(1.0), Circle(2.0)),
 collide(Circle(1.0), Rectangle(2.0, 3.0)),
 collide(Rectangle(1.0, 1.0), Rectangle(2.0, 2.0))]
# Each line is a METHOD, chosen from BOTH types. No ladder, no visitor, and adding a
# type adds methods without touching the ones above.

#%% ...but symmetry is NOT free — try the pair we didn't define
# We wrote collide(::Circle, ::Rectangle). We never wrote the mirror image. Run this:
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    println("MethodError, as promised:\n  ", sprint(showerror, e)[1:min(end, 120)], " …")
end
# Julia does NOT guess that collide is commutative — nothing says a physical collision
# is symmetric, so it refuses rather than assume. You say it explicitly, in one line:
collide(a::Rectangle, b::Circle) = collide(b, a)      # symmetry, declared once
collide(Rectangle(1.0, 1.0), Circle(2.0))
# The lesson isn't "Julia is annoying": it's that dispatch does exactly what you WROTE.
# Which is also why `methods(collide)` is an honest, complete list of what's supported.

#%% 4. Extending without editing — the "expression problem"
# Add a NEW operation on the existing types, without touching their definitions:
perimeter(c::Circle)    = 2π * c.r
perimeter(r::Rectangle) = 2 * (r.L + r.w)

perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])

#%% ...and add a TYPE, without touching existing code — a new struct + its methods
#   struct Triangle <: Shape
#       b::Float64
#       h::Float64
#   end
#   area(t::Triangle) = t.b * t.h / 2
#
#              add a TYPE            add an OPERATION
#   OOP        easy (new subclass)   EDIT EVERY class
#   dispatch   easy (new struct)     easy (new function)
#
# OOP makes new types cheap but new operations invasive; multiple dispatch makes
# BOTH cheap — on the EXTENSIBILITY axis.
#
# ⚠ Honest footnote, because "Julia solves the expression problem" is a claim you'll
# see repeated and it is too strong. Wadler's formulation also demands STATIC TYPE
# SAFETY / exhaustiveness: the compiler should reject the program if a case is
# missing. Julia guarantees nothing of the sort — define Triangle without an `area`
# method and you get a MethodError AT RUN TIME (you can even see `throw_methoderror`
# sitting in the generated IR), exactly like the collide pothole above. Add to that:
# method AMBIGUITIES (two methods equally specific → error), and TYPE PIRACY (extending
# someone else's function on someone else's types — legal, and a great way to break a
# package you never imported). So: dispatch wins the extensibility axis convincingly;
# it does NOT hand you compile-time exhaustiveness. Say "solves the extensibility half".

#%% 5. Why this matters for speed
# Not only elegant design — multiple dispatch is EXACTLY what makes the type
# specialization of the compilation module possible. When you call area(Circle(2.0)),
# Julia knows the argument is a Circle, picks the matching method, and COMPILES a
# specialized native version for that type. Same mechanism as f(3.0) vs f(3):
# dispatch on the types, then a specialized compilation.
#
# The decisive point is WHEN the type is known — and note the CONDITION, which is the
# part usually left out:
#   Python obj.method()  → at RUN time, by lookup, every call — unspecialized
#   Julia  area(x)       → at COMPILE time, *WHEN THE TYPE IS INFERABLE*: the method is
#                          chosen, inlined, and the lookup disappears.
#
# ⚠ When it is NOT inferable — abstract container, type instability — Julia falls back
# to RUN-TIME dispatch, EXACTLY like Python. That's the sum_global() of the compilation
# module, and it's our own section 2: `area.([Circle, Rectangle, Circle])` builds a
# Vector{Shape}, so THAT demo dispatches dynamically and we measured it ~7-10× slower.
# CONCRETE TYPES are what buy the speed. Dispatch is the mechanism; inference is the
# precondition. Multiple dispatch + JIT + inferable types = GENERIC and FAST.
#
# `methods(f)` lists every method Julia knows for a generic function:
methods(area)

#%% 6. Summary
# Where the method lives   Python: inside the object (obj.method())
#                          Julia:  outside the types, generic function
# Chosen from              Python: the type of obj (ONE — singledispatch too)
#                          Julia:  the types of ALL arguments
# Add a type               easy / easy
# Add an operation          Python: edit every class (or monkey-patch / singledispatch)
#                          Julia:  add a function, types untouched
# Two-argument dispatch     Python: isinstance ladder, or the Visitor pattern
#                          Julia:  just another method
# Link to speed             Julia: it's the ENGINE of specialization — BUT only when
#                          the types are inferable (Vector{Shape} → run-time lookup,
#                          measured ~7-10× slower in section 2)
#
# TAKEAWAY: multiple dispatch isn't just style — it's THE mechanism that lets Julia
# generate specialized native code, and the one thing you must feed it is CONCRETE
# TYPES. We'll see it again when the SAME `*` picks the GPU version all by itself,
# just because its arguments are CuArray.

#%% What's next
# On to parallelism on the CPU: THREADS, and a classic trap — the parallel sum that
# returns the wrong answer.
