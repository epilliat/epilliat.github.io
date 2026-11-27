# Julia for Python users — read the demonstrations; exercises in Python. ENSAI 3A.
# ⚠ KEEP IN SYNC with `pluto/0_julia_basics.jl`. Cells: `#%%` (Alt+Enter in VS Code).
# Sections 1–2 are the 15-minute path; 3 is reference.

#%% 1. Monte-Carlo π (the course's running example)
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

#%% 2. Reading the loop
# In count_in_disk above: function … end defines a function; return hits returns the count.
# 1:n includes both endpoints; _ is unused; ^ means power; <= also works for ≤.
# end closes each for/if block; indentation improves readability.

#%% Broadcasting — the dot
xs = [1.0, 4.0, 9.0]
(sqrt.(xs), xs .+ 1)   # NumPy: np.sqrt(xs), xs + 1

#%% Macros start with @
@time count_in_disk(10^7)   # run the expression and report execution time

#%% Python → Julia
# def f(x): …      → function f(x) … end
# range(n)         → 1:n for n repetitions (different loop values)
# x**2             → x^2
# v[0] / v[-1]     → v[1] / v[end]
# np.sqrt(xs)      → sqrt.(xs)
# Return to count_in_disk: why does 4 * count_in_disk(n) / n estimate π?

#%% 3. Reference — Variables & types
x = 3
y = 2.5
name = "Ada"
(x, y, name, typeof(x), typeof(y), typeof(name))

#%% Numbers
(10 / 4, 10 ÷ 4, -7 ÷ 2, fld(-7, 2), 10 % 4, 2^10, sqrt(2))   # ÷ truncates toward zero; Python // floors (different for negatives). Try 2**10.

#%% Unicode & juxtaposition
α = 0.1                                    # \alpha + TAB
(α, 2α, 2π, 3 ≤ 4, 2 ∈ [1, 2, 3])          # \le, \in

#%% Strings
a, b = 3, 4
greeting = "Hello, $name"
sum_str  = "$a + $b = $(a + b)"
joined   = "foo" * "bar"                   # try "foo" + "bar"
(greeting, sum_str, joined, 'a', typeof('a'))

#%% Arrays: 1-based, slices include both ends
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

#%% Functions and keyword arguments
function area_rect(L, w)
    return L * w
end
square(x) = x^2
cube = x -> x^3
greet(who; punct = "!") = "Hi $who$punct"
(area_rect(3, 4), square(5), cube(2), greet("Ada"), greet("Bob"; punct = "?"))

#%% Broadcasting and mutation
square.(xs)
@. xs^2 + 1          # dot each operation; chained dots fuse into one loop
xs .= sqrt.(xs)      # update xs
sort!(xs)            # ! convention: modifies its argument

#%% Structs define data types
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
