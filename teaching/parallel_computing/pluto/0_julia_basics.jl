### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ d0000000-0000-4a00-8000-000000000001
md"""
# Julia for Python users
### A short syntax bridge — before the performance story

**ENSAI 3A — Julia as the test bench, Python as the point of comparison**

---

You already know how to program: you are fluent in **Python**. Julia will feel familiar — it is
**dynamic**, **high-level**, has a **REPL**, and you almost never write a type. This module is a
**20-minute bridge**: the handful of things that are *different*, so the later modules read smoothly.

> 🧭 **No performance here, and no parallelism** — that starts in the next module (compilation &
> types). Here we only learn to *read and write* Julia. Everything below runs as pure Julia, no
> packages to install.

The differences that actually trip people up:

1. arrays are **1-based** and stored **column-major**;
2. blocks close with **`end`**, not indentation;
3. element-wise math uses the **dot** (broadcasting), not a library;
4. behaviour lives in **functions**, not inside classes.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000002
md"""
## Reading a Pluto notebook (30 seconds)

This is a **Pluto** notebook, not Jupyter. Two things to know:

- It is **reactive**: change a cell and every cell that depends on it re-runs automatically. A
  consequence — **each variable is defined in exactly one cell** (no re-assigning `x` in two cells).
- **One expression per cell.** To group several statements, wrap them in `begin … end` (they share
  the global scope) or in `let … end` (a local scope — the demo cells below use `let` so their names
  stay private and don't collide).

Click the **eye 👁** to the left of a cell to show/hide its code.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000003
md"""
## Variables & basic types

Same as Python: assign with `=`, no type declaration, types are inferred. `typeof` is Julia's
`type(...)`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000004
let
    x = 3          # Int64  — a whole number
    y = 2.5        # Float64
    name = "Ada"   # String
    (x, y, name, typeof(x), typeof(y), typeof(name))
end

# ╔═╡ d0000000-0000-4a00-8000-000000000005
let
    (10 / 4,     # 2.5  — `/` is ALWAYS true division → Float64 (like Python 3)
     10 ÷ 4,     # 2    — integer division (type \div then TAB); same as div(10, 4)
     10 % 4,     # 2    — remainder
     2^10,       # 1024 — power is `^`, NOT `**`
     sqrt(2))    # functions are called f(x), no method on the number
end

# ╔═╡ d0000000-0000-4a00-8000-000000000006
md"""
## Strings & interpolation

Double quotes for strings; **single quotes are a `Char`** (`'a'` is one character, not a string).
Interpolate with `\$`, and `\$(...)` for a whole expression — just like Python f-strings, minus the
`f` prefix. One surprise: **`*` concatenates** strings (not `+`).
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000007
let
    a, b = 3, 4
    name = "Ada"
    greeting = "Hello, $name"           # $name interpolates the variable
    sum_str  = "$a + $b = $(a + b)"     # $(...) → interpolate an expression
    joined   = "foo" * "bar"            # * concatenates (NOT +)
    (greeting, sum_str, joined, 'a', typeof('a'))
end

# ╔═╡ d0000000-0000-4a00-8000-000000000008
md"""
## Blocks end with `end`, not indentation

Julia does **not** use indentation to delimit blocks (indent for readability, but it's ignored).
Every `for`, `while`, `if`, `function`, `struct`, `let`, `begin` is closed by a matching **`end`**.
No colons `:` after the header either.

```julia
for i in 1:3
    println(i)
end            # ← this closes the loop
```
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000009
md"""
## Conditionals

`if / elseif / else / end` (note **`elseif`**, one word). The ternary is `cond ? a : b`.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000000a
let
    grade(x) =
        if x ≥ 16
            "excellent"
        elseif x ≥ 10      # one word: elseif
            "pass"
        else
            "fail"
        end
    sign_word(x) = x ≥ 0 ? "non-negative" : "negative"   # ternary  cond ? a : b
    (grade(17), grade(11), grade(4), sign_word(-2))
end

# ╔═╡ d0000000-0000-4a00-8000-00000000000b
md"""
## Functions — three ways to write them

The last expression is returned, so **`return` is optional**. Keyword arguments come after a `;`
and may have defaults. A trailing **`!`** in a name is a convention meaning *"mutates its argument"*
(`push!`, `sort!`) — it's just a naming habit, not syntax.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000000c
let
    # 1) long form
    function area_rect(L, w)
        return L * w          # `return` optional — the last expression is the result
    end
    # 2) short "assignment" form — great for one-liners
    square(x) = x^2
    # 3) anonymous (lambda):  x -> ...
    cube = x -> x^3
    # keyword args after `;`, with a default
    greet(name; punct = "!") = "Hi $name$punct"

    (area_rect(3, 4), square(5), cube(2), greet("Ada"), greet("Bob"; punct = "?"))
end

# ╔═╡ d0000000-0000-4a00-8000-00000000000d
md"""
## Loops

`for x in collection … end`. A numeric range like `1:5` is the loop counter. Handy helpers:
`enumerate` (index + value, **starting at 1**), `zip`, `eachindex`.

> 🚀 Unlike Python, **writing an explicit loop here is not a sin** — a Julia loop runs at C speed.
> *Why* is exactly the subject of the next module (compilation & types). For now, just write loops.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000000e
let
    total = 0
    for i in 1:5              # range 1,2,3,4,5 — BOTH ends included
        total += i
    end
    words = ["a", "b", "c"]
    labelled = [(i, w) for (i, w) in enumerate(words)]   # enumerate starts at 1
    (total, labelled)
end

# ╔═╡ d0000000-0000-4a00-8000-00000000000f
md"""
## Arrays — the big one 🎯

The difference that bites Python programmers most:

- **Indexing starts at 1**, not 0. `v[1]` is the first element.
- `end` inside `[ ]` is the last index: `v[end]`, `v[end-1]`.
- Slices and ranges **include both ends**: `v[2:4]` is elements 2, 3, 4.
- A `Vector` is written with commas `[1, 2, 3]`. In a **matrix**, spaces separate columns and `;`
  starts a new row: `[1 2 3; 4 5 6]`, indexed `M[row, col]`.
- Matrices are stored **column-major** (columns are contiguous) — the opposite of NumPy's default.
  It won't matter until the memory module, but it's why the first index is the "fast" one.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000010
let
    v = [10, 20, 30, 40, 50]
    (v[1],          # FIRST element — indexing starts at 1
     v[end],        # last element  (`end` is a keyword inside [ ])
     v[end-1],      # second to last
     v[2:4],        # slice 2..4, both ends included
     length(v))
end

# ╔═╡ d0000000-0000-4a00-8000-000000000011
let
    r  = 1:2:9                            # start:step:stop → 1,3,5,7,9
    sq = [i^2 for i in 1:5]               # comprehension → [1,4,9,16,25]
    ev = [i for i in 1:10 if iseven(i)]  # comprehension with a filter
    M  = [1 2 3; 4 5 6]                   # 2×3 matrix
    (collect(r), sq, ev, size(M), M[2, 3])   # M[row, col] → 6
end

# ╔═╡ d0000000-0000-4a00-8000-000000000012
md"""
## Broadcasting — the dot `.`

This replaces both NumPy's vectorised operators *and* element-wise list comprehensions. Put a **dot**
on a function or operator and it applies **element by element**: `sqrt.(x)`, `x .+ 1`, `x .* 2`. The
macro **`@.`** puts a dot on *every* operation in an expression.

Where in NumPy you'd write `np.sqrt(x)` and rely on it being vectorised, in Julia you make the
element-wise intent explicit with the dot — and Julia **fuses** several dotted operations into a
single loop, with no temporary arrays.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000013
let
    x = [1.0, 4.0, 9.0]
    y = @. x^2 + 1        # @. broadcasts EVERY operation: (x^2 + 1) element-wise
    (sqrt.(x),            # element-wise sqrt
     x .+ 1,              # add 1 to each
     x .* 2,              # double each
     y)
end

# ╔═╡ d0000000-0000-4a00-8000-000000000014
md"""
## Tuples, named tuples, dicts

- **Tuple**: fixed-size, ordered, may mix types — `(1, "two", 3.0)`. Destructure with `a, b, c = t`.
- **Named tuple**: `(x = 10, y = 20)`, fields read as `nt.x`.
- **Dict**: `Dict("a" => 1, "b" => 2)` — pairs use `=>` (not `:` as in Python).
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000015
let
    t  = (1, "two", 3.0)                    # tuple, mixed types
    a, b, c = t                             # destructuring
    nt = (x = 10, y = 20)                   # named tuple
    d  = Dict("apple" => 3, "pear" => 5)    # dict with => pairs
    (t[1], b, nt.x, d["apple"], haskey(d, "pear"))
end

# ╔═╡ d0000000-0000-4a00-8000-000000000016
md"""
## Structs instead of classes

Julia has **no classes**. You group data in a `struct`, and — this is the key idea of the *next
language module* — the **behaviour lives in functions outside the type**, not in methods inside it.
There is no `self`.

- A `struct` is **immutable by default**: fields can't be reassigned after construction.
- Use `mutable struct` when you need to change fields.
- The default constructor takes the fields in order — no `__init__`, no `self`, no `new`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000017
begin
    struct Point            # immutable by default
        x::Float64
        y::Float64
    end

    mutable struct Counter  # mutable: fields CAN be reassigned
        n::Int
    end

    md"Types `Point` and `Counter` defined ✓"
end

# ╔═╡ d0000000-0000-4a00-8000-000000000018
let
    p = Point(1.0, 2.0)     # default constructor from the fields — no `self`
    c = Counter(0)
    c.n += 1                # allowed: Counter is mutable
    # p.x = 5.0  would ERROR — Point is immutable
    (p.x, p.y, c.n)
end

# ╔═╡ d0000000-0000-4a00-8000-000000000019
md"""
## Unicode & math notation

A small delight: identifiers can be **Unicode**. Type a LaTeX name then **TAB** in the editor —
`\\alpha`+TAB gives `α`, `\\le`+TAB gives `≤`, `\\in`+TAB gives `∈`. `π` is built in. And a number
placed right before a name **multiplies** by juxtaposition: `2π` means `2 * π`, `2x` means `2 * x`.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001a
let
    α = 0.1                          # type \alpha then TAB
    radius = 2.0
    circumference = 2π * radius      # π built in; 2π = 2 * π (juxtaposition)
    (α, π, circumference, 3 ≤ 4, 2 ∈ [1, 2, 3])   # ≤ is \le , ∈ is \in
end

# ╔═╡ d0000000-0000-4a00-8000-00000000001b
md"""
## Packages

The standard library ships many modules you load with `using`:

```julia
using LinearAlgebra     # dot, norm, matrix factorisations
using Statistics        # mean, std, ...
using Random            # rand, shuffle, seeds
```

To add an external package, use the built-in package manager `Pkg`:

```julia
using Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools
```

`import X` also works (then you call `X.f`); `using X` brings the exported names into scope directly.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001c
md"""
## Cheat-sheet — Python → Julia

| Topic | Python | Julia |
|---|---|---|
| First array element | `v[0]` | `v[1]` — **1-based** |
| Last element | `v[-1]` | `v[end]` |
| Slice (both ends) | `v[1:4]` (excl. 4) | `v[2:4]` (**incl.** 4) |
| Block delimiter | indentation + `:` | keywords, closed by **`end`** |
| Else-if | `elif` | `elseif` |
| Power | `2 ** 10` | `2 ^ 10` |
| True division | `/` | `/` (same) |
| Integer division | `//` | `÷` or `div(a, b)` |
| String concat | `"a" + "b"` | `"a" * "b"` |
| f-string | `f"{x}"` | `"\$x"` (no `f`) |
| Print a line | `print(x)` | `println(x)` |
| Object method | `obj.method()` | `method(obj)` — **verb outside** |
| Class | `class C:` (has `self`) | `struct C` (**immutable**, no `self`) |
| Element-wise | NumPy `np.f(x)` | `f.(x)` — **the dot** |
| Copy vs alias | `b = a[:]` to copy | `b = a` **aliases**; use `copy(a)` |
| Absent value | `None` | `nothing` (and `missing` for data) |
| Comment | `#` | `#` (same) |
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001d
md"""
## Putting it together

A tiny function that uses almost everything above — a loop, `rand`, a condition, `≤`, juxtaposition
— and quietly previews the **red thread of the course**: estimating **π by Monte-Carlo**. Throw `n`
random points into the unit square; the fraction landing inside the quarter disk is about `π/4`.

*(No timing here — that's the next module. Just read the syntax.)*
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001e
begin
    "Count how many of `n` random points land inside the unit quarter-disk."
    function count_in_disk(n)
        inside = 0
        for _ in 1:n
            x, y = rand(), rand()      # rand() → a Float64 in [0, 1)
            if x^2 + y^2 ≤ 1           # inside the quarter disk?
                inside += 1
            end
        end
        return inside
    end

    n = 100_000
    hits = count_in_disk(n)
    pi_estimate = 4 * hits / n         # fraction inside ≈ π/4, so ×4
    (hits, pi_estimate)
end

# ╔═╡ d0000000-0000-4a00-8000-00000000001f
md"""
### What's next

You can now read and write Julia. On to the first real question of the course:

> **Why can the same computation be tens or hundreds of times faster** depending on how it's
> written? That's the module on **compilation & types** — where those "ordinary" loops turn out to
> run at C speed.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "71853c6197a6a7f222db0f1978c7cb232b87c5ee"

[deps]
"""

# ╔═╡ Cell order:
# ╟─d0000000-0000-4a00-8000-000000000001
# ╟─d0000000-0000-4a00-8000-000000000002
# ╟─d0000000-0000-4a00-8000-000000000003
# ╠═d0000000-0000-4a00-8000-000000000004
# ╠═d0000000-0000-4a00-8000-000000000005
# ╟─d0000000-0000-4a00-8000-000000000006
# ╠═d0000000-0000-4a00-8000-000000000007
# ╟─d0000000-0000-4a00-8000-000000000008
# ╟─d0000000-0000-4a00-8000-000000000009
# ╠═d0000000-0000-4a00-8000-00000000000a
# ╟─d0000000-0000-4a00-8000-00000000000b
# ╠═d0000000-0000-4a00-8000-00000000000c
# ╟─d0000000-0000-4a00-8000-00000000000d
# ╠═d0000000-0000-4a00-8000-00000000000e
# ╟─d0000000-0000-4a00-8000-00000000000f
# ╠═d0000000-0000-4a00-8000-000000000010
# ╠═d0000000-0000-4a00-8000-000000000011
# ╟─d0000000-0000-4a00-8000-000000000012
# ╠═d0000000-0000-4a00-8000-000000000013
# ╟─d0000000-0000-4a00-8000-000000000014
# ╠═d0000000-0000-4a00-8000-000000000015
# ╟─d0000000-0000-4a00-8000-000000000016
# ╠═d0000000-0000-4a00-8000-000000000017
# ╠═d0000000-0000-4a00-8000-000000000018
# ╟─d0000000-0000-4a00-8000-000000000019
# ╠═d0000000-0000-4a00-8000-00000000001a
# ╟─d0000000-0000-4a00-8000-00000000001b
# ╟─d0000000-0000-4a00-8000-00000000001c
# ╟─d0000000-0000-4a00-8000-00000000001d
# ╠═d0000000-0000-4a00-8000-00000000001e
# ╟─d0000000-0000-4a00-8000-00000000001f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
