### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ d0000000-0000-4a00-8000-000000000001
md"""
# Julia for Python users

Read the demonstrations in Julia; exercises in Python.
The 15-minute introduction follows one Monte-Carlo loop, its syntax, and two tools for later demonstrations.
The remaining cells are a syntax reference.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000002
md"""
## Reading a Pluto notebook (30 seconds)

- It is **reactive**: change a cell and every cell that depends on it re-runs. So **each variable
  is defined in exactly one cell**.
- **One expression per cell**: group statements with `begin … end` (global names) or `let … end`
  (local names — most demo cells below, so they don't collide).
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000003
md"""
## 3. Reference — Variables & types

Assign with `=`, no declaration. `typeof` is Python's `type`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000004
begin
    x = 3
    y = 2.5
    name = "Ada"
    (x, y, name, typeof(x), typeof(y), typeof(name))
end

# ╔═╡ 88eba8a3-d9a5-4bda-9a9b-70ddf9539e04
md"""
## Numbers

`÷` truncates toward zero; Python `//` floors. For integers, `fld(a, b)` uses floor division. Power is `^`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000005
(10 / 4, 10 ÷ 4, -7 ÷ 2, fld(-7, 2), 10 % 4, 2^10, sqrt(2))   # ÷ truncates toward zero; Python // floors (different for negatives). Try 2**10.

# ╔═╡ d0000000-0000-4a00-8000-000000000019
md"""
## Unicode & juxtaposition

LaTeX name + **TAB** gives `α`, `≤`, `∈`. A number before a name multiplies: `2π` is `2 * π`.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001a
let
    α = 0.1                                # \alpha + TAB
    (α, 2α, 2π, 3 ≤ 4, 2 ∈ [1, 2, 3])      # \le, \in
end

# ╔═╡ d0000000-0000-4a00-8000-000000000006
md"""
## Strings

Interpolate with `\$name` / `\$(expr)` (no `f` prefix). `'a'` is a `Char`. Which operator
concatenates? Try `"foo" + "bar"`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000007
let
    a, b = 3, 4
    greeting = "Hello, $name"
    sum_str  = "$a + $b = $(a + b)"
    joined   = "foo" * "bar"                   # try "foo" + "bar"
    (greeting, sum_str, joined, 'a', typeof('a'))
end

# ╔═╡ d0000000-0000-4a00-8000-00000000000f
md"""
## Arrays: 1-based, slices include both ends

`v[1]` is the first element, `v[end]` the last, `v[2:4]` is elements 2, 3 **and** 4. Try `v[0]`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000010
begin
    v = [10, 20, 30, 40, 50]
    (v[1], v[end], v[end-1], v[2:4], length(v))   # try v[0]
end

# ╔═╡ e7d7c618-df19-4c50-b667-9b1d967dcfbe
md"""
## Ranges, comprehensions, matrices

`start:step:stop`; in a matrix, spaces separate columns and `;` rows; index `M[row, col]`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000011
let
    r  = 1:2:9
    sq = [i^2 for i in 1:5]
    ev = [i for i in 1:10 if iseven(i)]
    M  = [1 2 3; 4 5 6]
    (collect(r), sq, ev, size(M), M[2, 3])
end

# ╔═╡ d0000000-0000-4a00-8000-000000000014
md"""
## Tuples, named tuples, dicts

Dict pairs use `=>`, not `:`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000015
let
    t  = (1, "two", 3.0)
    p, q, s = t
    nt = (x = 10, y = 20)
    d  = Dict("apple" => 3, "pear" => 5)
    (t[1], q, nt.x, d["apple"], haskey(d, "pear"))
end

# ╔═╡ d0000000-0000-4a00-8000-000000000008
md"""
## 2. Reading the loop

```julia
for _ in 1:n                 # repeat n times
    x, y = rand(), rand()    # sample a uniform point
    if x^2 + y^2 ≤ 1         # ^ means power; <= also works
        hits += 1
    end
end
```

`function … end` defines the function; `return hits` returns its count.
`1:n` includes both endpoints; `_` is unused. Indent for readability; `end` closes the block.
"""


# ╔═╡ d0000000-0000-4a00-8000-000000000009
md"""
## Conditionals

`elseif` is one word; the ternary is `cond ? a : b`.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000000a
let
    grade(x) =
        if x ≥ 16
            "excellent"
        elseif x ≥ 10
            "pass"
        else
            "fail"
        end
    sign_word(x) = x ≥ 0 ? "non-negative" : "negative"
    (grade(17), grade(11), grade(4), sign_word(-2))
end

# ╔═╡ d0000000-0000-4a00-8000-00000000000d
md"""
## Loops

What is the first index printed?
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000000e
let
    words = ["a", "b", "c"]
    for (i, w) in enumerate(words)
        println(i, " ", w)
    end
    [(i, w) for (i, w) in enumerate(words)]
end

# ╔═╡ 2ac7e0ea-96d0-4ce5-901c-b16d16d8617a
md"""
## `let … end`: a local scope

The last expression is the block's value.
"""

# ╔═╡ c67cda2c-5d47-41bb-9d6c-aedafbf62960
let
    acc = zero(eltype(v))    # a zero of v's element type (Int here)
    for i in eachindex(v)    # v's valid indices (1:5 here)
        acc += v[i]
    end
    acc                      # the last expression is the block's value
end

# ╔═╡ d0000000-0000-4a00-8000-00000000000b
md"""
## Functions and keyword arguments

`return` is optional (last expression). Keyword arguments come after `;`.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000000c
let
    function area_rect(L, w)
        return L * w
    end
    square(x) = x^2
    cube = x -> x^3
    greet(who; punct = "!") = "Hi $who$punct"
    (area_rect(3, 4), square(5), cube(2), greet("Ada"), greet("Bob"; punct = "?"))
end

# ╔═╡ d0000000-0000-4a00-8000-000000000012
md"""
## Broadcasting — the dot

NumPy `np.sqrt(xs)` corresponds to `sqrt.(xs)`; NumPy `xs + 1` corresponds to `xs .+ 1`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000013
let
    xs = [1.0, 4.0, 9.0]
    (sqrt.(xs), xs .+ 1)
end

# ╔═╡ d0000000-0000-4a00-8000-000000000016
md"""
## Structs define data types

No classes, no `self`: data in a `struct` (immutable by default), behaviour in functions outside.
Try `pt.x = 5.0`.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000017
begin
    struct Point
        x::Float64
        y::Float64
    end
    mutable struct Counter
        n::Int
    end
end

# ╔═╡ d0000000-0000-4a00-8000-000000000018
let
    pt = Point(1.0, 2.0)
    c  = Counter(0)
    c.n += 1
    (pt.x, pt.y, c.n)                          # try pt.x = 5.0
end

# ╔═╡ d0000000-0000-4a00-8000-00000000001b
md"""
## Packages

```julia
using Statistics            # stdlib: just `using`
using Pkg; Pkg.add("BenchmarkTools"); using BenchmarkTools
```
(In Pluto, `using X` alone installs the package for this notebook.)
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001d
md"""
## 1. Monte-Carlo π (the course's running example)

`n` random points in the unit square: the fraction inside the quarter-disk is ≈ `π/4`.
"""

# ╔═╡ d0000000-0000-4a00-8000-00000000001e
begin
    "Count how many of `n` random points land inside the unit quarter-disk."
    function count_in_disk(n)
        hits = 0
        for _ in 1:n
            x, y = rand(), rand()
            if x^2 + y^2 ≤ 1
                hits += 1
            end
        end
        return hits
    end
    n = 100_000
    hits = count_in_disk(n)
    pi_estimate = 4 * hits / n
    (hits, pi_estimate)
end

# ╔═╡ d0000000-0000-4a00-8000-00000000001f
md"""
## Macros start with `@`

`@time` runs an expression and reports its execution time. Later demonstrations use other macros.
"""

# ╔═╡ d0000000-0000-4a00-8000-000000000020
@time count_in_disk(10^7)

# ╔═╡ d0000000-0000-4a00-8000-00000000001c
md"""
## Python → Julia

| Python | Julia |
|---|---|
| `def f(x): …` | `function f(x) … end` |
| `for _ in range(n):` | `for _ in 1:n … end` |
| `x**2` | `x^2` |
| `v[0]` / `v[-1]` | `v[1]` / `v[end]` |
| `np.sqrt(xs)` | `sqrt.(xs)` |

Return to the loop: why does `4 * count_in_disk(n) / n` estimate π?
"""

# ╔═╡ dfa73de0-59a6-48ca-ad82-3f5428bc8761
md"""
## Broadcasting and mutation

Dotted operations apply element by element and fuse into one loop.
The `!` naming convention indicates a function that modifies an argument.
"""

# ╔═╡ dfa73de0-59a6-48ca-ad82-3f5428bc8762
let
    xs = [1.0, 4.0, 9.0]
    square(x) = x^2
    squares = square.(xs)
    shifted = @. xs^2 + 1
    xs .= sqrt.(xs)
    sort!(xs)
    (squares, shifted, xs)
end

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
# ╟─d0000000-0000-4a00-8000-00000000001d
# ╠═d0000000-0000-4a00-8000-00000000001e
# ╟─d0000000-0000-4a00-8000-000000000008
# ╟─d0000000-0000-4a00-8000-000000000012
# ╠═d0000000-0000-4a00-8000-000000000013
# ╟─d0000000-0000-4a00-8000-00000000001f
# ╠═d0000000-0000-4a00-8000-000000000020
# ╟─d0000000-0000-4a00-8000-00000000001c
# ╟─d0000000-0000-4a00-8000-000000000003
# ╠═d0000000-0000-4a00-8000-000000000004
# ╟─88eba8a3-d9a5-4bda-9a9b-70ddf9539e04
# ╠═d0000000-0000-4a00-8000-000000000005
# ╟─d0000000-0000-4a00-8000-000000000019
# ╠═d0000000-0000-4a00-8000-00000000001a
# ╟─d0000000-0000-4a00-8000-000000000006
# ╠═d0000000-0000-4a00-8000-000000000007
# ╟─d0000000-0000-4a00-8000-00000000000f
# ╠═d0000000-0000-4a00-8000-000000000010
# ╟─e7d7c618-df19-4c50-b667-9b1d967dcfbe
# ╠═d0000000-0000-4a00-8000-000000000011
# ╟─d0000000-0000-4a00-8000-000000000014
# ╠═d0000000-0000-4a00-8000-000000000015
# ╟─d0000000-0000-4a00-8000-000000000009
# ╠═d0000000-0000-4a00-8000-00000000000a
# ╟─d0000000-0000-4a00-8000-00000000000d
# ╠═d0000000-0000-4a00-8000-00000000000e
# ╟─2ac7e0ea-96d0-4ce5-901c-b16d16d8617a
# ╠═c67cda2c-5d47-41bb-9d6c-aedafbf62960
# ╟─d0000000-0000-4a00-8000-00000000000b
# ╠═d0000000-0000-4a00-8000-00000000000c
# ╟─dfa73de0-59a6-48ca-ad82-3f5428bc8761
# ╠═dfa73de0-59a6-48ca-ad82-3f5428bc8762
# ╟─d0000000-0000-4a00-8000-000000000016
# ╠═d0000000-0000-4a00-8000-000000000017
# ╠═d0000000-0000-4a00-8000-000000000018
# ╟─d0000000-0000-4a00-8000-00000000001b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
