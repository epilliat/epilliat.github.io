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
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001d
begin
    using BenchmarkTools   # @btime — section 2 measures what dispatch actually costs
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
## 1. OOP: the verb hangs on the right

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
	This is genuinely the right idea. Now read the name: **single** dispatch. It chooses on the type of **one** argument — the first. That limitation isn't an accident of the library; it's the whole point of **section 3**, where one argument stops being enough.

	So the interesting question is **not** *"can Python do it?"* (it can, awkwardly) but **"what does the language make the default, and what does the compiler get to assume?"** Keep that question in mind — section 5 answers it with a benchmark.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000004
md"""
## 2. Julia: the verb goes first, and sees the types

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

# ╔═╡ d2000000-0000-4a00-8000-000000000007
# the . broadcasts `area` over each element — the right method is picked per type
area.([Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)])

# ╔═╡ d2000000-0000-4a00-8000-000000000011
md"""
### ⚠️ Look at the type of that array before you believe the happy story

That demo is pretty. It is also, exactly as written, **the counter-example to this module's thesis**. Ask what container it actually built:
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000012
shapes = [Circle(2.0), Rectangle(3.0, 4.0), Circle(1.0)]

# ╔═╡ d2000000-0000-4a00-8000-000000000013
typeof(shapes)                     # → Vector{Shape} … an ABSTRACT element type!

# ╔═╡ d2000000-0000-4a00-8000-000000000014
isconcretetype(eltype(shapes))     # → false. Julia CANNOT know what area() will hit.

# ╔═╡ d2000000-0000-4a00-8000-000000000015
md"""
Mixing `Circle`s and `Rectangle`s forces the common supertype: **`Shape`**. So for *this* array the method **cannot** be chosen at compile time — Julia must look it up **at run time, per element**, exactly like Python.

**Our showcase demo does dynamic dispatch.** Don't take my word for it — measure. Same data, same answer; the only difference is the *type of the container*.
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
    print("Vector{Shape}  (abstract) : "); @btime total($mixed)
    print("Vector{Circle} (concrete) : "); @btime total($concrete)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000019
md"""
Roughly **1 µs vs 100 ns** — the concrete container wins by about an **order of magnitude** (7–15× here, depending on machine load; the concrete version is so fast the ratio moves). Identical values, identical result, **zero allocations on both sides**. The only difference is whether the compiler could know the type **in advance**.

Ask the compiler what it produced. In the abstract case the IR carries a **dynamic call to `Main.area`** — the lookup survived into the running code:
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001a
@code_typed total(mixed)      # ← look for the dynamic `Main.area` call

# ╔═╡ d2000000-0000-4a00-8000-00000000001b
@code_typed total(concrete)   # ← area resolved and inlined: the lookup is GONE

# ╔═╡ d2000000-0000-4a00-8000-00000000001c
md"""
!!! tip "This is the whole module, in one measurement"
	Dispatch buys you speed **only when the types are concrete**. An abstract container gives you the flexibility **and** the Python price tag. It links straight back to `@code_warntype` from the compilation module: same disease, same symptom.

	**Practical rule.** A `Vector{Shape}` is not a bug — sometimes you genuinely need a mixed bag. But *know what it costs*, and prefer a concrete container (or a small `Union`, which Julia can **union-split**) in a hot loop.
"""

# ╔═╡ d2000000-0000-4a00-8000-000000000008
md"""
## 3. The difference that counts: all arguments, not just one

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

### …but symmetry is *not* free

We wrote `collide(::Circle, ::Rectangle)`. We never wrote the mirror image. Try it:
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000001f
try
    collide(Rectangle(1.0, 1.0), Circle(2.0))
catch e
    sprint(showerror, e)      # → MethodError: no method matching collide(::Rectangle, ::Circle)
end

# ╔═╡ d2000000-0000-4a00-8000-000000000020
md"""
Julia does **not** guess that `collide` is commutative — nothing says a physical collision is symmetric, so it **refuses rather than assume**. You state it explicitly, in one line:

```julia
collide(a::Rectangle, b::Circle) = collide(b, a)      # symmetry, declared once
```

The lesson isn't *"Julia is annoying"*: it's that **dispatch does exactly what you wrote**. Which is also why `methods(collide)` is an honest, complete list of what's actually supported.
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000a
md"""
## 4. Extending without editing — the "expression problem"

**Add an operation** on the existing types, without touching their definitions:
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

# ╔═╡ d2000000-0000-4a00-8000-00000000000d
md"""
## 5. Why this matters for speed

This isn't only elegant design — **multiple dispatch is exactly what makes the type specialization of the compilation module possible.**

When you call `area(Circle(2.0))`, Julia knows the argument is a `Circle`, picks the matching method, and **compiles a specialized native version** for that type. Same mechanism as `f(3.0)` vs `f(3)`: dispatch on the types, then a specialized compilation.

The decisive point is *when* the type is known — and note the **condition**, which is the part usually left out:

| | method resolved… |
|---|---|
| Python `obj.method()` | **at run time**, by lookup, every call — unspecialized |
| Julia `area(x)` | **at compile time — *when the type is inferable***: method chosen, inlined, lookup gone |

!!! warning "And when it is *not* inferable, Julia dispatches at run time — exactly like Python"
	Abstract container, type instability → Julia falls back to a **run-time lookup**. That's the `sum_global()` of the compilation module. And it's **our own section 2**: `area.([Circle, Rectangle, Circle])` builds a `Vector{Shape}`, so *that* demo dispatches dynamically — we measured it about an **order of magnitude** slower.

	**Concrete types are what buy the speed.** Dispatch is the mechanism; **inference is the precondition.**

> Multiple dispatch **+** JIT **+ inferable types** = functions that are **generic** *and* **fast** — the central idea of the language.

`methods(f)` lists every method Julia knows for a generic function:
"""

# ╔═╡ d2000000-0000-4a00-8000-00000000000e
methods(area)

# ╔═╡ d2000000-0000-4a00-8000-00000000000f
md"""
## 6. Summary

| Idea | Python (OOP) | Julia (dispatch) |
|---|---|---|
| Where the method lives | **inside** the object (`obj.method()`) | **outside** the types, generic function |
| Chosen from | the type of `obj` (**one** — `singledispatch` too) | the types of **all** arguments |
| Add a type | easy | easy |
| Add an operation | edit every class (or monkey-patch / `singledispatch`) | add a function, types untouched |
| Two-argument dispatch | `isinstance` ladder, or the Visitor pattern | just another method |
| Link to speed | — | **it's the engine** of specialization — *but only when the types are inferable* |

> **Takeaway.** Multiple dispatch isn't just a matter of style: it's *the* mechanism that lets Julia generate specialized native code — and the one thing you must feed it is **concrete types** (section 2 measured what happens otherwise). We'll see it again when the *same* `*` picks the **GPU** version all by itself, just because its arguments are `CuArray`.
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
# ╠═d2000000-0000-4a00-8000-000000000007
# ╟─d2000000-0000-4a00-8000-000000000011
# ╠═d2000000-0000-4a00-8000-000000000012
# ╠═d2000000-0000-4a00-8000-000000000013
# ╠═d2000000-0000-4a00-8000-000000000014
# ╟─d2000000-0000-4a00-8000-000000000015
# ╠═d2000000-0000-4a00-8000-000000000016
# ╟─d2000000-0000-4a00-8000-000000000017
# ╠═d2000000-0000-4a00-8000-000000000018
# ╟─d2000000-0000-4a00-8000-000000000019
# ╠═d2000000-0000-4a00-8000-00000000001a
# ╠═d2000000-0000-4a00-8000-00000000001b
# ╟─d2000000-0000-4a00-8000-00000000001c
# ╟─d2000000-0000-4a00-8000-000000000008
# ╠═d2000000-0000-4a00-8000-000000000009
# ╟─d2000000-0000-4a00-8000-00000000001e
# ╠═d2000000-0000-4a00-8000-00000000001f
# ╟─d2000000-0000-4a00-8000-000000000020
# ╟─d2000000-0000-4a00-8000-00000000000a
# ╠═d2000000-0000-4a00-8000-00000000000b
# ╟─d2000000-0000-4a00-8000-00000000000c
# ╟─d2000000-0000-4a00-8000-00000000000d
# ╠═d2000000-0000-4a00-8000-00000000000e
# ╟─d2000000-0000-4a00-8000-00000000000f
# ╟─d2000000-0000-4a00-8000-000000000010
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
