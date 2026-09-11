# =============================================================================
# Julia for Python users — a short syntax bridge (before the performance story)
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC with `pluto/0_julia_basics.jl` — same lesson, two formats.
# Cells are delimited by `#%%` (Alt+Enter in VS Code). Pure Julia, nothing to install.
# No performance and no parallelism here — that starts in the next module.
#
# The four things that actually trip Python programmers up:
#   1. arrays are 1-based and stored column-major;
#   2. blocks close with `end`, not indentation;
#   3. element-wise math uses the dot (broadcasting), not a library;
#   4. behaviour lives in functions, not inside classes.
# =============================================================================

#%% Variables & basic types
# Same as Python: assign with `=`, no type declaration, types are inferred.
# `typeof` is Julia's `type(...)`.
x = 3          # Int64  — a whole number
y = 2.5        # Float64
name = "Ada"   # String
(x, y, name, typeof(x), typeof(y), typeof(name))

#%% Numbers: division, power, remainder
(10 / 4,     # 2.5  — `/` is ALWAYS true division → Float64 (like Python 3)
 10 ÷ 4,     # 2    — integer division (type \div then TAB); same as div(10, 4)
 10 % 4,     # 2    — remainder
 2^10,       # 1024 — power is `^`, NOT `**`
 sqrt(2))    # functions are called f(x), no method on the number

#%% Strings & interpolation
# Double quotes = String; single quotes = a Char ('a' is ONE character).
# Interpolate with $, and $(...) for an expression — like an f-string, no `f`.
# Surprise: `*` concatenates strings (not `+`).
a, b = 3, 4
greeting = "Hello, $name"           # $name interpolates the variable
sum_str  = "$a + $b = $(a + b)"     # $(...) interpolates an expression
joined   = "foo" * "bar"            # * concatenates (NOT +)
(greeting, sum_str, joined, 'a', typeof('a'))

#%% Blocks end with `end`, not indentation
# No `:` after the header; indent for readability only. Every for/while/if/
# function/struct/let/begin is closed by a matching `end`.
for i in 1:3
    println(i)
end            # ← this closes the loop


#%% Functions — three ways to write them
# `return` is optional (the last expression is returned); keyword args come after `;`.
# A trailing `!` (push!, sort!) is a CONVENTION for "mutates its argument".
function area_rect(L, w)      # 1) long form
    return L * w
end
square(x) = x^2               # 2) short "assignment" form — great for one-liners
cube = x -> x^3               # 3) anonymous (lambda):  x -> ...
greet(who; punct = "!") = "Hi $who$punct"   # keyword arg with a default
(area_rect(3, 4), square(5), cube(2), greet("Ada"), greet("Bob"; punct = "?"))


#%% Conditionals
# if / elseif / else / end  (note: `elseif`, one word). Ternary: cond ? a : b
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



#%% Loops
# `for x in collection … end`; helpers: enumerate (from 1), zip, eachindex.
# Unlike Python, an explicit loop is NOT a sin here — it runs at C speed (next module).
total = 0
for i in 1:5              # range 1,2,3,4,5 — BOTH ends included
    global total += i    # `global` needed to touch a global from a loop in a script
end
words = ["a", "b", "c"]
labelled = [(i, w) for (i, w) in enumerate(words)]   # enumerate starts at 1
(total, labelled)

#%% Arrays
# - Indexing starts at 1 (v[1] is the first element).
# - `end` inside [ ] is the last index: v[end], v[end-1].
# - Slices/ranges INCLUDE both ends: v[2:4] is elements 2,3,4.
# - Vector: commas [1,2,3]. Matrix: spaces = columns, ; = new row, M[row, col].
# - Matrices are stored COLUMN-major (opposite of NumPy) — matters in the memory
#   module; it's why the first index is the "fast" one.
v = [10, 20, 30, 40, 50]
(v[1],          # FIRST element — indexing starts at 1
 v[end],        # last element  (`end` is a keyword inside [ ])
 v[end-1],      # second to last
 v[2:4],        # slice 2..4, both ends included
 length(v))

#%% Ranges, comprehensions, matrices
r  = 1:2:9                            # start:step:stop → 1,3,5,7,9
sq = [i^2 for i in 1:5]               # comprehension → [1,4,9,16,25]
ev = [i for i in 1:10 if iseven(i)]  # comprehension with a filter
M  = [1 2 3; 4 5 6]                   # 2×3 matrix
(collect(r), sq, ev, size(M), M[2, 3])   # M[row, col] → 6

#%% Broadcasting — the dot `.`
# Replaces NumPy's vectorised operators. A dot on any function/operator applies it
# element-wise; `@.` dots EVERY operation. Chained dots fuse into ONE loop — no
# temporary arrays (that matters in the memory module).
xs = [1.0, 4.0, 9.0]
z = @. xs^2 + 1        # broadcasts (x^2 + 1) element-wise
(sqrt.(xs),            # element-wise sqrt
 xs .+ 1,              # add 1 to each
 xs .* 2,              # double each
 z)

#%% Tuples, named tuples, dicts
# Tuple: fixed-size, ordered, mixed types. Named tuple: fields via .name.
# Dict: pairs use => (not `:` as in Python).
t  = (1, "two", 3.0)                    # tuple, mixed types
p, q, s = t                             # destructuring
nt = (x = 10, y = 20)                   # named tuple
d  = Dict("apple" => 3, "pear" => 5)    # dict with => pairs
(t[1], q, nt.x, d["apple"], haskey(d, "pear"))

#%% Structs instead of classes
# No classes, no `self`. Data in a `struct`, behaviour OUTSIDE in functions (the next
# language module). Immutable by default; `mutable struct` allows reassignment.
struct Point            # immutable by default
    x::Float64
    y::Float64
end
mutable struct Counter  # mutable: fields CAN be reassigned
    n::Int
end
pt = Point(1.0, 2.0)    # default constructor — no __init__, no self, no new
c  = Counter(0)
c.n += 1                # allowed: Counter is mutable
# pt.x = 5.0  would ERROR — Point is immutable
(pt.x, pt.y, c.n)

#%% Unicode & math notation
# Identifiers can be Unicode: type a LaTeX name then TAB (\alpha→α, \le→≤, \in→∈).
# π is built in. A number right before a name multiplies: 2π = 2*π, 2x = 2*x.
α = 0.1                          # type \alpha then TAB
radius = 2.0
circumference = 2π * radius      # π built in; 2π = 2 * π (juxtaposition)
(α, π, circumference, 3 ≤ 4, 2 ∈ [1, 2, 3])   # ≤ is \le , ∈ is \in

#%% Packages
# Standard library modules load with `using`:
#     using LinearAlgebra   # dot, norm, factorisations
#     using Statistics      # mean, std, ...
#     using Random          # rand, shuffle, seeds
# External packages via the built-in manager:
#     using Pkg
#     Pkg.add("BenchmarkTools")
#     using BenchmarkTools
# `import X` also works (call X.f); `using X` brings exported names into scope.

#%% Cheat-sheet — Python → Julia
# First / last element     v[0] / v[-1]      →  v[1] / v[end]      (1-based)
# Slice (both ends)        v[1:4] excl. 4    →  v[2:4] incl. 4
# Block delimiter          indent + ':'      →  keywords, closed by `end`
# Else-if                  elif              →  elseif
# Power / integer div      2**10 / //        →  2^10 / ÷ (or div)
# String concat / f-string "a"+"b" / f"{x}"  →  "a"*"b" / "$x"
# Object method            obj.method()      →  method(obj)   (verb outside)
# Class                    class C (self)    →  struct C (immutable, no self)
# Element-wise             np.f(x)           →  f.(x)   (the dot)
# Copy vs alias            b = a[:]          →  b = a aliases; use copy(a)
# Absent value             None              →  nothing (and `missing` for data)

#%% Putting it together — the course red thread (Monte-Carlo π)
# Throw n random points into the unit square: the fraction inside the quarter disk
# is ≈ π/4. No timing yet — just read the syntax.
"Count how many of `n` random points land inside the unit quarter-disk."
function count_in_disk(n)
    inside = 0
    for _ in 1:n
        x, y = rand(), rand()      # rand() → a Float64 in [0, 1)
        if x^2 + y^2 <= 1           # inside the quarter disk?
            inside += 1
        end
    end
    return inside
end
n = 100_000
hits = count_in_disk(n)
pi_estimate = 4 * hits / n         # fraction inside ≈ π/4, so ×4
(hits, pi_estimate)

#%% What's next
# You can now read and write Julia. Next: WHY the same computation can be tens or
# hundreds of times faster depending on how it is written — compilation & types,
# where those "ordinary" loops turn out to run at C speed.
