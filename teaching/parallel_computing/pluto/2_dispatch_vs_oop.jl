### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ d2000000-0000-4a00-8000-000000000001
md"""
# Multiple dispatch vs OOP
### Where the verb lives: left vs right of the data

**ENSAI 3A — Julia as the test bench, Python as the point of comparison**

---

In the compilation module we saw *why* the same computation can be 100× faster: compiling and **specializing on the types** removes the per-operation overhead.

Here we look at the **language mechanism** that makes that specialization possible — **multiple dispatch** — by contrasting it head-on with the **object-oriented** (OOP) style you already know from Python.

> 🧭 This is a **language point**, not parallelism yet. Threads come right after — but dispatch is what will make them elegant (and what will later move a computation onto the GPU without changing the code).

**For each cell: predict the output, then run it.**
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001d
begin
    using BenchmarkTools   # @btime / @benchmark — sections 3 and 4 measure what dispatch and types cost
    md"Packages loaded ✓"
end

# ╔═╡ d2000000-0000-4a00-8000-000000000002
md"""
## Where does the verb go?

The *same* idea — "the area of a shape" — written two ways:

| | call | the verb is… |
|---|---|---|
| **OOP** (Python) | `shape.area()` | **on the right**, after the dot |
| **dispatch** (Julia) | `area(shape)` | **on the left**, prefix |

- OOP: the method is **owned by the object** — it lives *inside* the class.
- Julia: `area` is a **generic function**, living **outside** the types.

The receiver **left of the dot** decides the OOP method. The **verb on the left** decides the Julia method — from the types of **all** its arguments.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000003
md"""
## 1. One argument: one function, one method per type

### OOP: the verb hangs on the right

You know Python's object-oriented style: the method belongs to the object, called as `object.method(...)`. The choice is made on **one** object — the one to the left of the dot.

```python
class Shape:
    def area(self): ...

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return 3.14159 * self.r**2

class Rectangle(Shape):
    def __init__(self, L, w): self.L, self.w = L, w
    def area(self): return self.L * self.w

Circle(2.0).area()      # dispatch on the object (left of the dot)
```

The method is **locked inside** the class. To add an operation (say `perimeter`), you must **edit every class**.

!!! warning "🐍 \"No! Python can do that too\" — and you're right. Let's say it out loud now"
	The honest answer is what makes the rest of this module land. Two counter-arguments, and what each really costs:

	**(a) Monkey-patching** — bolt the method on from outside:
	```python
	Circle.perimeter = lambda self: 2 * 3.14159 * self.r
	```
	It works. It also mutates a class other code shares, at run time, invisibly — and it fails outright on builtins (`int`, `list`) and on anything with `__slots__` or a C extension type. It's an escape hatch, not a design.

	**(b) `functools.singledispatch`** — a real generic function, verb on the left:
	```python
	@singledispatch
	def area(s): raise NotImplementedError
	@area.register
	def _(s: Circle): return 3.14159 * s.r**2
	```
	This is genuinely the right idea. Now read the name: **single** dispatch. It chooses on the type of **one** argument — the first. That limitation isn't an accident of the library; it's the whole point of **the next section**, where one argument stops being enough.

	So the interesting question is **not** *"can Python do it?"* (it can, awkwardly) but **"what does the language make the default, and what does the compiler get to assume?"** Keep that question in mind — the speed section answers it with a benchmark.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000004
md"""
### Julia: the verb goes first, and sees the types

In Julia we **separate the data (types) from the operations (functions)**. First the types…
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
begin
    c = Circle(0.5)
    area(c)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000007
# the . broadcasts `area` over each element — the right method is picked per type
begin
    shapes = [Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)]
    area.(shapes)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000008
md"""
## 2. Two arguments: dispatch on the PAIR of types

Because the verb is on the left, it can look at **every** argument — not only the first.

A function `collide(a, b)` can choose a different method depending on the **pair of types** `(a, b)` — this is **multiple dispatch**. With `a.collide(b)`, only `a` has a say; the second argument gets none.

### 🐍 Feel the pain first

This is what you write in Python — and `singledispatch` does **not** save you here, because it dispatches on the first argument only, so the *second* type has to be re-discovered by hand:

```python
class Circle(Shape):
    def collide(self, other):
        if   isinstance(other, Circle):    return "two circles"
        elif isinstance(other, Rectangle): return "circle and rectangle"
        elif isinstance(other, Triangle):  return "circle and triangle"
        else: raise TypeError(f"Circle vs {type(other)}?")

class Rectangle(Shape):
    def collide(self, other):
        if   isinstance(other, Circle):    return "circle and rectangle"
        elif isinstance(other, Rectangle): return "two rectangles"
        ...                                # the SAME ladder again, per class
```

**The `isinstance` ladder is the tell**: the language dispatched on *one* type, so **you** hand-write the dispatch on the other. With N shapes that's N classes × N branches — and adding `Triangle` means **editing every existing class**, the very thing OOP was supposed to make easy.

> The classic escape is the **Visitor pattern**: an entire design pattern that exists only to fake double dispatch in a single-dispatch language.

Now the Julia version — each line is a **method**, chosen from **both** types:
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

# ╔═╡ d2000000-0000-4a00-8000-00000000001e
md"""
No ladder, no visitor — and adding a type adds methods **without touching the ones above**.

### The pair we did not define: what happens?

We wrote `collide(::Circle, ::Rectangle)`. We never wrote the mirror image. **Bet first**, then look:
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001f
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    sprint(showerror, e)      # → MethodError: no method matching collide(::Rectangle, ::Circle)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000020
md"""
Julia does **not** guess that `collide` is commutative — nothing says a physical collision is symmetric, so it **refuses rather than assume**.

### Add the missing pair in one line

The next cell is **disabled** on purpose (Pluto is reactive: once the method exists, the error above is gone). Enable it (cell menu `⋯` → *Enable cell*) and watch the cell above re-run and now answer `"circle and rectangle"`.
"""

# ╔═╡ e663a7e4-bb42-4b8f-a61c-c121ba409b20
# ╠═╡ disabled = true
#=╠═╡
begin
    collide(a::Rectangle, b::Circle) = collide(b, a)      # symmetry, declared once
    collide(Rectangle(1.0, 1.0), Circle(2.0))
end
  ╠═╡ =#

# ╔═╡ 909b04ea-913c-4ee6-afb2-de8624244214
md"""
The lesson isn't *"Julia is annoying"*: it's that **dispatch does exactly what you wrote**. Which is also why `methods(collide)` is an honest, complete list of what's actually supported.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000a
md"""
### A new operation, without touching the types

The "expression problem": **add an operation** on the existing types, without touching their definitions:
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000b
# a NEW operation on the existing types — their definitions stay untouched
begin
    perimeter(c::Circle)    = 2π * c.r
    perimeter(r::Rectangle) = 2 * (r.L + r.w)

    perimeter.([Circle(2.0), Rectangle(3.0, 4.0)])
end

# ╔═╡ d2000000-0000-4a00-8000-00000000000c
md"""
**Add a type**, without touching existing code — a new `struct` and its own methods:

```julia
struct Triangle <: Shape
    b::Float64
    h::Float64
end
area(t::Triangle) = t.b * t.h / 2
```

| | add a **type** | add an **operation** |
|---|---|---|
| OOP | easy (new subclass) | **edit every class** |
| dispatch | easy (new `struct`) | easy (new function) |

OOP makes new types cheap but new operations invasive; multiple dispatch makes **both** cheap — **on the extensibility axis**.

!!! warning "Honest footnote: \"Julia solves the expression problem\" is too strong"
	You'll see that claim repeated. Wadler's formulation also demands **static type safety / exhaustiveness**: the compiler should *reject* the program when a case is missing. Julia guarantees nothing of the sort — define `Triangle` without an `area` method and you get a `MethodError` **at run time** (you can even find `throw_methoderror` sitting in the generated IR), exactly like the `collide` pothole above.

	Add to that **method ambiguities** (two methods equally specific → error) and **type piracy** (extending someone else's function on someone else's types — legal, and a great way to break a package you never imported).

	So: dispatch wins the **extensibility** axis convincingly; it does **not** hand you compile-time exhaustiveness. Say *"it solves the extensibility half"*.
"""

# ╔═╡ 33dc6971-3ba9-42e4-871b-491010f9d5d7
md"""
### Who knows which methods? How many for `area`?

`methods(f)` lists every method Julia knows for a generic function:
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000e
methods(area)

# ╔═╡ d2000000-0000-4a00-8000-00000000000d
md"""
## 3. Dispatch and speed

This isn't only elegant design — **multiple dispatch is exactly what makes the type specialization of the compilation module possible.**

When you call `area(Circle(2.0))`, Julia knows the argument is a `Circle`, picks the matching method, and **compiles a specialized native version** for that type. Same mechanism as `f(3.0)` vs `f(3)`: dispatch on the types, then a specialized compilation.

The decisive point is *when* the type is known — and note the **condition**, which is the part usually left out:

| | method resolved… |
|---|---|
| Python `obj.method()` | **at run time**, by lookup, every call — unspecialized |
| Julia `area(x)` | **at compile time — *when the type is inferable***: method chosen, inlined, lookup gone |

### What container did `shapes` build?

Our first demo, `area.(shapes)`, was pretty. Look at the type of that array before you believe the happy story:
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000013
typeof(shapes)

# ╔═╡ d2000000-0000-4a00-8000-000000000011
md"""
Is its element type **concrete**?
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000014
isconcretetype(eltype(shapes))

# ╔═╡ d2000000-0000-4a00-8000-000000000015
md"""
Mixing `Circle`s and `Rectangle`s forces the common supertype: **`Shape`** — an *abstract* element type. So for *this* array the method **cannot** be chosen at compile time — Julia must look it up **at run time, per element**, exactly like Python.

### Same data, two containers: predict which is faster, and by how much

Same data, same answer; the only difference is the *type of the container*.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000016
total(v) = sum(area, v)

# ╔═╡ d2000000-0000-4a00-8000-000000000017
begin
    radii    = rand(1000)
    concrete = [Circle(r) for r in radii]        # Vector{Circle} → concrete
    mixed    = Shape[Circle(r) for r in radii]   # Vector{Shape}  → abstract, SAME data
    @assert total(mixed) ≈ total(concrete)
    md"Same radii in both containers ✓ — only the container's type differs"
end

# ╔═╡ d2000000-0000-4a00-8000-000000000018
let
    print("Vector{Shape}  : "); @btime total($mixed)
    print("Vector{Circle} : "); @btime total($concrete)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000019
md"""
Roughly **1 µs vs 100 ns** — the concrete container wins by about an **order of magnitude** (7–15× here, depending on machine load; the concrete version is so fast the ratio moves). Identical values, identical result, **zero allocations on both sides**. The only difference is whether the compiler could know the type **in advance**.

### Why? Find the call to `area` in each
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001a
@code_typed total(mixed)

# ╔═╡ 006f4b78-17e9-44bd-995e-2539ff17f17b
md"""
…and here?
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001b
@code_typed total(concrete)

# ╔═╡ d2000000-0000-4a00-8000-00000000001c
md"""
In the abstract case the IR carries a **dynamic call to `Main.area`** — the lookup survived into the running code. In the concrete case `area` is resolved and inlined: the lookup is **gone**.

!!! tip "This is the whole module, in one measurement"
	Dispatch buys you speed **only when the types are concrete**. An abstract container gives you the flexibility **and** the Python price tag. **Concrete types are what buy the speed.** Dispatch is the mechanism; **inference is the precondition.**

	**Practical rule.** A `Vector{Shape}` is not a bug — sometimes you genuinely need a mixed bag. But *know what it costs*, and prefer a concrete container (or a small `Union`, which Julia can **union-split**) in a hot loop.
"""

# ╔═╡ 55f08765-0106-49a5-a7d7-0abe97f4a0c4
md"""
## 4. Type instability — same sum, global vs argument

The same disease has a more common form: an **untyped global variable**. The compiler cannot fix its type, so every function that reads it falls back to run-time lookups.

Two versions of the same sum — one reads a global, the other takes its array as an argument:
"""

# ╔═╡ 924a2d90-3322-4415-aa37-a919702b77ee
glob = rand(1000)

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
### Spot the red `Any`

`@code_warntype` colors in **red** the places where Julia doesn't know the type (`Any`) — code falling back to dynamic behavior, like Python.
"""

# ╔═╡ e5ba66a4-5eab-46f1-99ae-99ad4e5eba4d
@code_warntype sum_global()

# ╔═╡ f140228c-e890-4b33-a784-4c0209cde97d
md"""
…and here?
"""

# ╔═╡ f11b9f37-a293-4355-b9da-aa95bb5192a2
@code_warntype sum_arg(glob)

# ╔═╡ 1340c246-2259-4b37-afdb-787ed4ae7678
md"""
### Predict the gap, then measure
"""

# ╔═╡ 1c3ebba3-7520-4544-b579-de2c5f7f543b
@benchmark sum_global()

# ╔═╡ 5e4e8915-06a6-48ca-824f-64f3ee2645ae
@benchmark sum_arg($glob)

# ╔═╡ 814dc34f-c5a2-4e51-82ba-56e3a16fc55b
md"""
**Julia golden rule**: put the work in **functions** that receive their data **as arguments**. That's what enables type specialization — hence speed. A loop "at global scope" in a Julia script is a common mistake when coming from Python.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000f
md"""
## Summary

| Idea | Python (OOP) | Julia (dispatch) |
|---|---|---|
| Where the method lives | **inside** the object (`obj.method()`) | **outside** the types, generic function |
| Chosen from | the type of `obj` (**one** — `singledispatch` too) | the types of **all** arguments |
| Add a type | easy | easy |
| Add an operation | edit every class (or monkey-patch / `singledispatch`) | add a function, types untouched |
| Two-argument dispatch | `isinstance` ladder, or the Visitor pattern | just another method |
| Link to speed | — | **it's the engine** of specialization — *but only when the types are inferable* |

> **Takeaway.** Multiple dispatch isn't just a matter of style: it's *the* mechanism that lets Julia generate specialized native code — and the one thing you must feed it is **concrete types** (an abstract container or an untyped global costs you about an order of magnitude). We'll see it again when the *same* `*` picks the **GPU** version all by itself, just because its arguments are `CuArray`.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000010
md"""
### What's next

On to **parallelism on the CPU**: **threads**, and a classic trap — the **parallel sum that returns the wrong answer**.
"""

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
# ╟─d2000000-0000-4a00-8000-00000000001d
# ╟─d2000000-0000-4a00-8000-000000000002
# ╟─d2000000-0000-4a00-8000-000000000003
# ╟─d2000000-0000-4a00-8000-000000000004
# ╠═d2000000-0000-4a00-8000-000000000005
# ╠═d2000000-0000-4a00-8000-000000000006
# ╠═2b32054d-cdc1-449c-90c7-8aab98109770
# ╠═d2000000-0000-4a00-8000-000000000007
# ╟─d2000000-0000-4a00-8000-000000000008
# ╠═d2000000-0000-4a00-8000-000000000009
# ╟─d2000000-0000-4a00-8000-00000000001e
# ╠═d2000000-0000-4a00-8000-00000000001f
# ╟─d2000000-0000-4a00-8000-000000000020
# ╠═e663a7e4-bb42-4b8f-a61c-c121ba409b20
# ╟─909b04ea-913c-4ee6-afb2-de8624244214
# ╟─d2000000-0000-4a00-8000-00000000000a
# ╠═d2000000-0000-4a00-8000-00000000000b
# ╟─d2000000-0000-4a00-8000-00000000000c
# ╟─33dc6971-3ba9-42e4-871b-491010f9d5d7
# ╠═d2000000-0000-4a00-8000-00000000000e
# ╟─d2000000-0000-4a00-8000-00000000000d
# ╠═d2000000-0000-4a00-8000-000000000013
# ╟─d2000000-0000-4a00-8000-000000000011
# ╠═d2000000-0000-4a00-8000-000000000014
# ╟─d2000000-0000-4a00-8000-000000000015
# ╠═d2000000-0000-4a00-8000-000000000016
# ╟─d2000000-0000-4a00-8000-000000000017
# ╠═d2000000-0000-4a00-8000-000000000018
# ╟─d2000000-0000-4a00-8000-000000000019
# ╠═d2000000-0000-4a00-8000-00000000001a
# ╟─006f4b78-17e9-44bd-995e-2539ff17f17b
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
