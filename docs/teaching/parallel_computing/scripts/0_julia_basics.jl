# Julia for Python users — a syntax bridge (no performance yet). ENSAI 3A.
# ⚠ KEEP IN SYNC with `pluto/0_julia_basics.jl`. Cells: `#%%` (Alt+Enter in VS Code).
# For each cell: predict the output, then run it.

#%% 1. Values — Variables & types
x = 3
y = 2.5
name = "Ada"
(x, y, name, typeof(x), typeof(y), typeof(name))

#%% Numbers
(10 / 4, 10 ÷ 4, 10 % 4, 2^10, sqrt(2))   # ÷ is \div + TAB. Try 2**10.

#%% Unicode & juxtaposition
α = 0.1                                    # \alpha + TAB
(α, 2α, 2π, 3 ≤ 4, 2 ∈ [1, 2, 3])          # \le, \in

#%% Strings
a, b = 3, 4
greeting = "Hello, $name"
sum_str  = "$a + $b = $(a + b)"
joined   = "foo" * "bar"                   # try "foo" + "bar"
(greeting, sum_str, joined, 'a', typeof('a'))

#%% 2. Collections — Arrays: 1-based, slices include both ends
v = [10, 20, 30, 40, 50]
(v[1], v[end], v[end-1], v[2:4], length(v))   # try v[0]

#%% Ranges, comprehensions, matrices
r  = 1:2:9
sq = [i^2 for i in 1:5]
ev = [i for i in 1:10 if iseven(i)]
M  = [1 2 3; 4 5 6]
(collect(r), sq, ev, size(M), M[2, 3])

#%% Tuples, named tuples, dicts
t  = (1, "two", 3.0)
p, q, s = t
nt = (x = 10, y = 20)
d  = Dict("apple" => 3, "pear" => 5)
(t[1], q, nt.x, d["apple"], haskey(d, "pear"))

#%% 3. Control flow — Blocks close with `end`
for i in 1:3
    println(i)
end

#%% Conditionals
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

#%% Loops
# What is the first index printed?
words = ["a", "b", "c"]
for (i, w) in enumerate(words)
    println(i, " ", w)
end
[(i, w) for (i, w) in enumerate(words)]

#%% `let … end`: a local scope
let
    acc = zero(eltype(v))    # a zero of v's element type (Int here)
    for i in eachindex(v)    # v's valid indices (1:5 here)
        acc += v[i]
    end
    acc                      # the last expression is the block's value
end

#%% 4. Functions & types — Functions, three forms
function area_rect(L, w)
    return L * w
end
square(x) = x^2
cube = x -> x^3
greet(who; punct = "!") = "Hi $who$punct"
(area_rect(3, 4), square(5), cube(2), greet("Ada"), greet("Bob"; punct = "?"))

#%% Broadcasting — the dot
xs = [1.0, 4.0, 9.0]
(sqrt.(xs), xs .+ 1, @. xs^2 + 1)          # try sqrt(xs)

#%% Structs instead of classes
struct Point
    x::Float64
    y::Float64
end
mutable struct Counter
    n::Int
end
pt = Point(1.0, 2.0)
c  = Counter(0)
c.n += 1
(pt.x, pt.y, c.n)                          # try pt.x = 5.0
# Changing a struct's fields later needs a REPL restart.

#%% Packages
# using Statistics            # stdlib: just `using`
# using Pkg; Pkg.add("BenchmarkTools"); using BenchmarkTools

#%% 5. Monte-Carlo π (the course's running example)
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

#%% Cheat-sheet — Python → Julia
# v[0] / v[-1]        →  v[1] / v[end]
# v[1:4] (excl. 4)    →  v[2:4] (incl. 4)
# indent + ':' / elif →  `end` / elseif
# 2**10 / //          →  2^10 / ÷
# "a"+"b" / f"{x}"    →  "a"*"b" / "$x"
# np.f(x)             →  f.(x)
# obj.method()        →  method(obj)
# class C (self)      →  struct C (immutable, no self)
# b = a[:] (copy)     →  b = a aliases; copy(a)
# None                →  nothing (missing for data)
