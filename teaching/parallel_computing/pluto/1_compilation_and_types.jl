### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 86593a2f-41c2-464d-905a-70b368785d81
begin
    using BenchmarkTools   # @btime, @benchmark: reliable measurements
    using PlutoUI          # sliders, formatting
    using Printf
    using Profile          # @profile: the sampling profiler (section 7)
    using Random           # MersenneTwister (reproducible darts)
    md"Packages loaded ✓"
end

# ╔═╡ ab181e11-6516-4169-9189-fb4488f4c8d3
md"""
# Parallel computing — Interpreted vs compiled
## Why the same computation can be 100× faster

**ENSAI 3A — Julia as a test bench, Python as a point of comparison**

---

The thread of the whole course is one question:

> *Why can the same computation, mathematically identical, be tens or even hundreds of times faster depending on how it's written?*

Today we answer with three ingredients:

1. **Interpreted vs compiled** — what really happens between your code and the processor.
2. **Type specialization** — why Julia can be as fast as C.
3. **Measuring properly** — a performance number is only worth as much as its measurement protocol.

Parallelism (threads, GPU) comes in later modules. But you can't sensibly parallelize code whose sequential cost you don't already understand.
"""

# ╔═╡ e4c6629b-29ce-4e63-b529-04bf1020ee4f
md"""
## 0. Getting started

This cell installs and loads the packages we need. The **first run** may take a minute or two (download + precompilation): that's already an illustration of the course — Julia compiles.
"""

# ╔═╡ 7525bf04-f5c7-45aa-b826-642c15b68eaf
@btime 1+2

# ╔═╡ d4572c88-919a-4a54-9581-8f241edc1d7a
@benchmark 1+2

# ╔═╡ 9c279c3b-f2c3-4a98-8f21-b21190182e5b
md"""
## 1. Same computation, very different times

Let's start with the simplest possible demo: **summing a large vector of numbers**.

We'll write this computation three ways:

- a **pure Python loop** (which we'll simulate mentally / compare against);
- the **`numpy`** version;
- a **Julia loop**.

The computation is *exactly the same*. The times are not at all.
"""

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81176
# A vector of 10 million Float64
const N = 10_000_000

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81177
const v = rand(N)

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81178
summary(v)

# ╔═╡ 26289681-0477-4670-950f-8351be0fdd5a
# Sum via a hand-written Julia loop
function sum_loop(x)
    s = zero(eltype(x))
    for i in eachindex(x)
        s += x[i]
    end
    return s
end

# ╔═╡ da96898d-efc5-4e47-a73e-8e0a991bc0f8
sum_loop(v)   # check it works

# ╔═╡ bb0ace77-07a4-4504-80c1-edc793f3f899
md"""
### The Python point of comparison

In pure Python, the same computation would read:

```python
def sum_loop(x):
    s = 0.0
    for xi in x:
        s += xi
    return s
```

On 10 million elements, this Python loop typically takes **~1 second**. The Julia loop above takes a few **milliseconds** — a factor of about **100×**, for code that *looks the same line for line*.

`numpy.sum(x)`, on the other hand, is fast (~10 ms) — but because it **doesn't run in Python**: it calls a compiled C routine. We come back to this in section 5.

So the question becomes: **why is the Julia loop fast while the Python loop is slow?**
"""

# ╔═╡ 99808d58-8875-42ed-bd94-c7e06ab36c10
md"""
## 2. The warmup: the first call is special

The **very first call** of a function pays the **JIT compilation**; the following calls are the real cost. Run the next cell: the first `@time` is inflated (huge allocations, "compilation" time), the second is the real time.
"""

# ╔═╡ a8128f4d-9cdc-4cd2-8025-33c5935757e0
let
    g(x) = sum(abs2, x)        # new function, never compiled
    @time g(v)                 # 1st call: includes compilation
    @time g(v)                 # 2nd call: the real cost
end

# ╔═╡ 05891168-bf4c-42c6-86f1-5d038606e267
md"""
**The warmup, seen over several calls.** Let's time **each** call of a *fresh* function over the first iterations: the 1st pays the compilation (spike), then the time **decays** to the steady state.
"""

# ╔═╡ b1180cde-dd1e-4983-aae0-bd631034c0df
let
    # FRESH function (never compiled). Trick: we call it via an abstract container
    # Function[...] so the compilation falls INSIDE the 1st timed call
    # (otherwise Julia compiles it beforehand, before the loop, and the spike disappears).
    function fresh(x)
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]
        end
        return s
    end
    fb = Function[fresh]
    times = [@elapsed(fb[1](v)) for _ in 1:10]
    for k in eachindex(times)
        println(@sprintf("call %2d : %8.3f ms", k, times[k] * 1e3))
    end
end

# ╔═╡ 97acf5d7-5b90-4b3a-a2b9-ffdda2d75a52
md"""
The **spike** on the 1st call is the **JIT** compiling `fresh` for `Vector{Float64}`. **Rule of thumb**: always discard this 1st call, then measure the steady state (which is what `@btime` does).

So *what* is this compilation, exactly? That's the next section.
"""

# ╔═╡ 22d3648b-a71a-4299-8e1b-46f032a74046
md"""
## 3. Interpreted vs compiled: observing the compilation

The spike we just saw is Julia **compiling**. Three ways to get from source code to the processor:

- **interpreted** (CPython): the bytecode is re-examined at every operation → slow, no warmup;
- **ahead-of-time / AOT** (C, `numpy`): compiled before running → fast, no warmup;
- **just-in-time / JIT** (Julia): compiled on the 1st call, then native → fast *after* the warmup.

Julia lets you **inspect** every stage between source code and assembly. Let's take a deliberately trivial function so the output stays readable.
"""

# ╔═╡ a51ea6fa-9f15-453f-bcea-852c7e71a354
f(x) = 2x + 1

# ╔═╡ e8d202f2-8080-4e46-aa29-157c5a707c13
md"""
### Step 1 — the *lowered* code

Julia first rewrites your code into a canonical form of elementary instructions (SSA). It is still type-independent.
"""

# ╔═╡ 26118708-e6fa-4cdb-a5e9-7a72a376b84f
@code_lowered f(3.0)

# ╔═╡ c3c88228-8ec2-4a08-9384-7da546be08fb
md"""
### Step 2 — the typed code

This is the **decisive** step. Julia now knows the argument's type (`Float64`) and **propagates the types** through the whole function: each operation knows what it acts on.

**This is where "one compilation per type" happens.** The *same* function `f` produces different code depending on the argument type: `f(3.0)` (`Float64`) and `f(3)` (`Int`) trigger **two distinct compiled versions**, each specialized for its type. (Julia calls this a *specialization* / *method instance*: it compiles a new one on the **first** encounter of each type combination, then reuses it.)

Compare the two outputs:
"""

# ╔═╡ c14f0bba-a2a2-457d-a400-35224ec23b34
@code_typed f(3.0)

# ╔═╡ eb5c0a66-0e2e-4eba-9cc6-0b6f6acd4fb6
@code_typed f(3)

# ╔═╡ 3e4c87e0-e1aa-4593-8f71-a57b44ef14fe
md"""
### Step 3 — the assembly

We go all the way down to processor instructions. Per-type compilation becomes **directly observable**:

- `f(3.0)` (`Float64`) → **floating-point** instructions (`vmulsd`, `vaddsd`…);
- `f(3)` (`Int`) → **integer** instructions (`lea`, `add`…).

Two types → **two different machine codes**, a few lines each, no loop and no "interpreter". Proof, at the lowest level, that there really is a compiled version **per type**.
"""

# ╔═╡ cf7f8128-23b8-4c2e-a122-d2705f2c43c6
@code_native debuginfo=:none f(3.0)

# ╔═╡ d5000000-0000-4d00-9b00-00000000d001
@code_native debuginfo=:none f(3)

# ╔═╡ cb125025-5a35-4e36-bf13-7efc3e29abd4
md"""
### What to remember

- **Python (CPython)** keeps your code as *bytecode* and runs it through a big interpretation loop. Every `+` goes through the loop, on **boxed** objects allocated on the heap. This **cost per operation** is what penalizes the loop.
- **Julia** compiles, *on first use and for each type combination*, a specialized native version. Once compiled, `2x+1` on `Float64` is literally just two machine instructions.

That's the meaning of "Julia is compiled": not a compiler run by hand before execution, but **just-in-time (JIT)** compilation, triggered by the argument types.

!!! warning "Don't say \"Python can't specialize\" — it's false, and the truth is a better argument"
	Since **3.11** (PEP 659) CPython has a *specializing adaptive interpreter*: it watches the types flowing through each bytecode and rewrites it in place — a warmed-up `s += x` on floats really does become `BINARY_OP_ADD_FLOAT`, with no `__add__` lookup left. (`dis.dis()` hides this; you need `dis.get_instructions(f, adaptive=True)`.)

	The difference is **what gets specialized, and what survives**. CPython specializes **one bytecode** and keeps a **type guard on every execution**, still inside the interpreter, still on **boxed** heap objects. Julia specializes **the whole function**, ahead of the loop, **proves** the guard once and deletes it, and works on unboxed values in registers.

	CPython removed the method lookup — the cheapest part. That's why the ~100× survives PEP 659. The Python track measures this (`python/1`).

> **Key distinction — what is specialized, and does the check survive?**
> - **Python**: one instruction at a time, the guard is re-checked at *every* operation, inside the interpreter, boxed → **slow**.
> - **Julia/JIT**: the whole function, the type is known once, the check **disappears** from the hot code, unboxed → **fast**.
>
> This **compile-time vs run-time** opposition will return for **dispatch**, **threads** and the **GPU**.
"""

# ╔═╡ 656f12ed-0f30-498d-8462-f58bff519eda
md"""
### Going further: why Base `sum` beats our loop

`sum_loop` is already compiled native code. Yet Base `sum` is **faster** — *exactly the same computation*. To isolate the **computation** (not the memory), let's take a vector that **fits in cache**:
"""

# ╔═╡ 6f3bc084-05bc-4b17-98f6-5d306b03740e
const vc = rand(100_000)   # ~800 KB: fits in cache

# ╔═╡ c3c32bc0-9d67-46e2-bdbe-c7d3f84ee2f8
@btime sum_loop($vc)

# ╔═╡ 42fbb9f9-240e-487c-8f3d-ad839115f426
@btime sum($vc)

# ╔═╡ e8b6b134-eb4f-49f2-8f39-eb3c3f1365e9
md"""
`sum` is **~5× faster**. The reason is in the assembly. First **our** loop:
"""

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82804
@code_native debuginfo=:none sum_loop(vc)

# ╔═╡ ba010064-49f9-4872-a4fe-a42d2e754d75
md"""
We see **`vaddsd`**: *Scalar Double* = **one** float at a time, in **a single** accumulator (`xmm0`). Each `+` waits for the previous one → a single dependency chain (*latency-bound*).

Now look at `sum` itself — and brace yourself for a surprise:
"""

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82805
@code_native debuginfo=:none sum(vc)

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82806
md"""
!!! warning "No `vaddpd` here either! Don't stop reading now"
	If you scan this listing for the SIMD we promised, you won't find it: no `vaddpd`, no `ymm`, just a handful of `vaddsd`. The obvious conclusion — *"so `sum` isn't vectorized after all"* — is **wrong**. The work simply **isn't in this function**. Three things to spot:

	```asm
	cmp     rdx, 15
	jle     .LBB0_6                        ← N ≤ 15? handle it right here
	movabs  rax, offset j_mapreduce_impl   ← otherwise CALL the real kernel
	mov     ecx, 1024                      ← ...with a block size of 1024
	call    rax
	```

	So the `vaddsd` you *do* see belong to the **small-N path** (N ≤ 15, fully unrolled) — not to the hot loop at all. The **`N ≥ 16` threshold** and the **1024-element pairwise block** are literally readable in the machine code; both live in Base's `reduce.jl`.

	*Lesson beyond `sum`: `@code_native` shows you **one** function. When the work is behind a `call`, you have to follow it.*

Let's follow the call. **This** is where the SIMD lives:
"""

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82807
@code_native debuginfo=:none Base.mapreduce_impl(identity, Base.add_sum, vc, 1, length(vc), 1024)

# ╔═╡ ba010064-49f9-4872-a4fe-a42d2e754d76
md"""
There it is — **`vaddpd` on `ymm0..ymm3`**: *Packed Double* = **4 floats at once** (SIMD) × **4 independent accumulators** → ~16 doubles per iteration, 4 dependency chains in parallel (**ILP**). At the end they're folded together (`vaddpd ymm0, ymm1, ymm0` …).

Why isn't our loop allowed to do that? Floating-point addition is **not associative** (`(a+b)+c ≠ a+(b+c)`, due to rounding): without permission, the compiler **must** preserve the order. `sum` goes through a reduction that **allows reassociation**. We can grant the same permission to our loop with **`@simd`**:
"""

# ╔═╡ 8e0cdaff-257b-4836-8998-2cd5f0ef8044
function sum_loop_simd(x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)   # @simd: allows reassociation
        s += x[i]
    end
    return s
end

# ╔═╡ 0662912c-e7ab-4223-82d2-e2683543f99b
@btime sum_loop_simd($vc)

# ╔═╡ c62ea649-7bc8-4b64-b662-b874c41547f5
md"""
A single word (`@simd`) and the loop **vectorizes**: it catches up with (or beats) `sum`. So speed doesn't come from a "magic" language but from **what the compiler is allowed to do** — here, reassociate to exploit SIMD + instruction-level parallelism.

*(Reproducible demo — assembly + `@simd` benchmark: `experiences/sum_vs_size/reflection.jl`.)*
"""

# ╔═╡ fad9827d-f3db-43ae-b0bb-b58aca083e79
md"""
## 4. Pitfall #1: type instability

"Julia is fast" is false if the compiler **cannot** determine the types. The most common case: an **untyped global variable**.

Here is the same sum, but reading a global variable.
"""

# ╔═╡ fa9596a7-23bd-4675-998b-af232a7b08e8
# global variable (type not fixed from the compiler's point of view)
glob = rand(1000)

# ╔═╡ fa9596a7-23bd-4675-998b-af232a7b08e9
function sum_global()
    s = 0.0
    for i in eachindex(glob)   # glob is a global
        s += glob[i]
    end
    return s
end

# ╔═╡ 41165e5b-f988-4698-97d1-04ca43eb7964
function sum_arg(x)
    s = 0.0
    for i in eachindex(x)
        s += x[i]
    end
    return s
end

# ╔═╡ 761ce2e0-503b-48b1-9a5d-2b2cce7f4f45
md"""
The key tool is `@code_warntype`: it colors in **red** the places where Julia doesn't know the type (`Any`). Red = code falling back to dynamic behavior, comparable to Python.

Look at the global version (red / `Any`):
"""

# ╔═╡ ba6f002b-32d3-4327-a14e-ad049956e1ec
@code_warntype sum_global()

# ╔═╡ 8d4f26b4-4d42-4291-8c31-0891ba765a37
md"""
…then the version that takes its array **as an argument** (everything typed, no red):
"""

# ╔═╡ 0fcbe125-0551-42b2-93b7-2fe0f6e65bb2
@code_warntype sum_arg(glob)

# ╔═╡ bd7de916-7afd-4b63-81e3-2f82fc00b935
md"""
### The lesson, measured

Both do the same computation. Let's measure the gap.
"""

# ╔═╡ f53445b9-d198-4ad4-9414-140afdb6460c
@benchmark sum_global()

# ╔═╡ 20c2f0e8-b528-40cf-94b5-e308ae298137
@benchmark sum_arg($glob)

# ╔═╡ 46d41ce1-700e-41eb-8ae4-02a37802725b
md"""
**Julia golden rule**: put the work in **functions** that receive their data **as arguments**. That's what enables type specialization — hence speed. A loop "at global scope" in a Julia script is a common mistake when coming from Python.
"""

# ╔═╡ 260d7cdf-31b5-47d5-8ea5-ccc76e270dbe
md"""
## 5. Why numpy is fast (and what it hides)

Back to Python. `numpy.sum(x)` is ~100× faster than a Python loop. Why?

Because `numpy` **doesn't run the loop in Python**. It stores the data in a contiguous typed array, and delegates the sum to a **compiled C function** (numpy's `.so` / `.pyd` files, themselves from C). Python only *calls* that C.

Important consequence:

- As long as you stay in "vectorized" numpy operations (`x + y`, `x.sum()`, `x @ y`), you're fast.
- **As soon as you write a Python loop around numpy elements**, you pay the interpreter cost on every iteration, and it all collapses.

Julia doesn't have this boundary: the loop *and* the "vectorized" form are the same compiled language. This is the *two-language problem* — Python solves it by writing the fast parts in another language (C, Cython, Numba); Julia tries to eliminate it.

### What about Numba?

`@numba.njit` adds a JIT to Python: it compiles certain numeric functions to native code, and can make a Python loop almost as fast as Julia. But it's a **subset** of Python (limited types, not every Python object is supported), and you must trigger it explicitly. It's exactly Julia's mechanism… added on top of Python for a restricted domain. A good counterpoint: "JIT" isn't magic, it's *specialize by types then compile*.
"""

# ╔═╡ d02cd083-dea1-417f-a941-8eac616197a6
md"""
## 6. Measuring properly — the art of benchmarking

Everything above rests on numbers, and a wrong number is worse than no number. Two pitfalls:

- **Pitfall A — timing the compilation.** That's the **warmup** from section 2: the 1st call includes JIT compilation. Always **discard the 1st call**.
- **Pitfall B — measuring only once** (below).
"""

# ╔═╡ 46c146f1-d1e4-4797-87bc-2c03a0185daf
md"""
### Pitfall B — measuring only once

A single measurement is polluted by noise (OS, CPU frequency, cache). You must **repeat** and look at the distribution. That's the whole point of `BenchmarkTools`:

- `@btime`: shows the **minimum** (most representative of the "pure" cost), plus allocations.
- `@benchmark`: the full distribution (min / median / mean / max).

⚠️ **Always interpolate variables with `\$`** (`@btime f(\$x)`) so the variable isn't treated as a global — otherwise you measure the pitfall from section 4, not your function.
"""

# ╔═╡ ca49af93-0baa-47ee-a4ec-3849a1cacffb
@benchmark sum_loop($v)

# ╔═╡ d1000000-0000-4a00-8000-000000000001
md"""
## 7. Profiling — *where* does the time actually go?

`@btime` answers **"how long?"**. On a real script with several functions the question is **"where?"** — and you can't `@btime` every line.

`Profile` (stdlib) is a **sampling** profiler: it interrupts the program every few milliseconds and records the **call stack**. The cost is negligible and nothing needs instrumenting; a function's **sample count** is roughly its share of the time.

> ⚙️ Run this section with **one thread** (`julia --threads=1`, the default). With several threads the *idle* ones flood the report with `poptask`/`wait` frames — we come back to that in the threads module, where that noise becomes the actual lesson.

Here is a small **layered pipeline**. Which of the three costs the most? **Bet before you look.**
"""

# ╔═╡ d1000000-0000-4a00-8000-000000000002
begin
    clean(v)     = [x for x in v if x > 0.01]      # keep the useful values
    transform(v) = sqrt.(abs.(v))
    function score(v)                               # sin+cos+exp in a loop — surely THIS one?
        s = 0.0
        for x in v
            s += sin(x) * cos(x) * exp(-x)
        end
        return s
    end
    pipeline(v) = score(transform(clean(v)))
end

# ╔═╡ d1000000-0000-4a00-8000-000000000003
const pdata = rand(3_000_000)

# ╔═╡ d1000000-0000-4a00-8000-000000000004
# profile it — warm up first, never profile the compilation
let
    pipeline(pdata)
    Profile.clear()
    Profile.@profile for _ in 1:10
        pipeline(pdata)
    end
    Profile.print(format = :flat, sortedby = :count, mincount = 60)
end

# ╔═╡ d1000000-0000-4a00-8000-000000000005
md"""
### What the report says

The absolute numbers vary run to run — **the order is the lesson**:

| count | frame | |
|---|---|---|
| **620** | `clean` | ← **more than `score`**. Nobody bets on this one. |
| 602 | `push!` | ⎫ |
| 543 | `_growend!` | ⎬ all **under `clean`** — it isn't computing, it's **allocating** |
| 524 | `GenericMemory` | ⎭ |
| **490** | `score` | ← the "obvious" suspect, only **second** |
| 157 | `transform` | |

`[x for x in v if cond]` **cannot know the final size**, so it **grows** the array: reallocate + copy, again and again.

> The profiler pointed straight at a **one-line function that looks free** — and told us its cost is **memory, not arithmetic**.
"""

# ╔═╡ d1000000-0000-4a00-8000-000000000006
# the fix: allocate the full size ONCE, then resize down
clean_fast(v) = filter(>(0.01), v)

# ╔═╡ d1000000-0000-4a00-8000-000000000007
let
    @assert clean(pdata) == clean_fast(pdata)
    print("comprehension : "); @btime clean($pdata)
    print("filter        : "); @btime clean_fast($pdata)
end

# ╔═╡ d1000000-0000-4a00-8000-000000000008
md"""
```
comprehension : ~7 ms  (36 allocations: 80.48 MiB)
filter        : ~4 ms  ( 3 allocations: 22.89 MiB)
```

**~1.7× faster and 3.5× less memory, for one word changed** — and we'd never have looked there without the profiler.

### *"So it was all about the growth?"* — test that, don't assume it

If regrowing the array were the whole story, **pre-sizing** it should recover *all* the lost time. That hypothesis is one function away, so let's write it: the same `push!` loop, told the size upfront.
"""

# ╔═╡ d1000000-0000-4a00-8000-000000000009
function clean_sizehint(v)
    out = similar(v, 0)
    sizehint!(out, length(v))          # no more regrowing: reserve everything now
    for x in v
        x > 0.01 && push!(out, x)
    end
    out
end

# ╔═╡ d1000000-0000-4a00-8000-00000000000a
let
    @assert clean_sizehint(pdata) == clean_fast(pdata)
    print("filter          : "); @btime clean_fast($pdata)
    print("push!+sizehint! : "); @btime clean_sizehint($pdata)
end

# ╔═╡ d1000000-0000-4a00-8000-00000000000b
md"""
**3 allocations, 22.89 MiB — *exactly* `filter`'s allocations.** And yet it's still clearly slower. So growth was **not** the whole story.

Why? Read Base's `filter` ([`array.jl:2932`](https://github.com/JuliaLang/julia/blob/master/base/array.jl)) — it's worth the detour:

```julia
function filter(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{T}(undef, length(a))
    for ai in a
        @inbounds b[j] = ai                 # write ALWAYS, unconditionally
        j = ifelse(f(ai)::Bool, j+1, j)     # only the CURSOR is conditional
    end
    resize!(b, j-1); sizehint!(b, length(b)); b
end
```

There is **no branch in that loop**. It writes *every* element, then decides whether to keep it by moving `j` — with `ifelse`, which compiles to a **conditional move**, not a jump. Our `push!` loop branches on every element; with random data the predictor is wrong a good fraction of the time, and each miss costs ~15–20 cycles of pipeline flush.

So the honest accounting is:

| | explains |
|---|---|
| the **growth** | the **memory** (80.48 → 22.89 MiB), and part of the time |
| the **branch** | the rest — *same allocations, still slower* |

!!! tip "This is the module's thesis, one level down"
	Speed is **what the machine must do per element**. Not the language. Not even the allocations alone.

### The loop that matters

> `@btime` says **how long** · the profiler says **where** · the allocations say **why**.
>
> **profile → diagnose → fix → re-measure.** That's the method of this whole course — and what you're graded on, whatever the language.
"""

# ╔═╡ 49ea0717-5e4a-44df-a895-11b9d55c747f
md"""
## 8. Application — Monte-Carlo π

Let's put it all into practice on a classic: estimate π by drawing random points in the square $[0,1]^2$ and counting the fraction that falls in the quarter disk.

$$\pi \approx 4 \times \frac{\#\{x^2+y^2 \le 1\}}{n}$$

We'll meet it again later in the course (sequential → multi-thread → GPU).
"""

# ╔═╡ 760cd379-4178-4e7f-9695-a1d709b36969
function estimate_pi(n)
    hits = 0
    for _ in 1:n
        x = rand(); y = rand()
        if x*x + y*y <= 1.0
            hits += 1
        end
    end
    return 4 * hits / n
end

# ╔═╡ 8adcfe41-1d0d-475a-968c-5a5a971a142b
estimate_pi(10_000_000)

# ╔═╡ 855d77ed-ca4b-407a-b225-2be1a2c499f5
md"""
**Exercises.** Answer in the following cells (edit the code, Pluto re-evaluates automatically).

**Q1.** Measure `estimate_pi` with `@btime` for `n = 10⁷`. How long? How many allocations? (Hint: `rand()` with no argument doesn't allocate — that's intentional.)
"""

# ╔═╡ 3cb6763c-dc00-4c09-94b5-289f773dc460
# Q1: your measurement here
@btime estimate_pi(10_000_000)

# ╔═╡ f15bd89d-47f6-4c87-a3d9-17078b722f1e
md"""
**Q2.** Check type stability with `@code_warntype estimate_pi(1000)`. You'll find `Body::Float64`, `hits::Int64` — everything concrete, and **no red at all**. Two things are worth understanding here rather than pattern-matching:

1. **Why is the result always `Float64`, never `Int`** — whatever the branches do? (Look again at what `/` does in Julia. Check yourself with `Base.return_types(estimate_pi, (Int,))`.)
2. **The one highlighted line is not a bug.** What is `@_3::Union{Nothing, Tuple{Int64, Int64}}`, and why does it appear in *every* `for` loop you will ever write?

!!! warning "Learning to ignore *that* yellow matters as much as spotting the real red"
	Section 4 taught you "red = `Any` = bad". If you go hunting for red here, you'll land on this yellow `Union` instead and conclude that an ordinary `for` loop is type-unstable — **the exact opposite of the lesson**.

	`@_3` is the **iterator protocol**: `iterate` returns either a `(value, state)` tuple or `nothing` when the range is exhausted. That `Union` is *benign*, the compiler **union-splits** it, and it costs nothing. Compare with `sum_global()` above, where `Any` really does show up.

	As for `/`: it is **always true division** in Julia (the Julia-basics module!), so `4 * hits / n` is `Float64` even when `hits` is `0`. The `if` changes the **value** of `hits`, never its **type**.
"""

# ╔═╡ 654b39b4-b592-4066-b519-0783a704f637
# Q2: your analysis here
@code_warntype estimate_pi(1000)

# ╔═╡ 654b39b4-b592-4066-b519-0783a704f638
Base.return_types(estimate_pi, (Int,))   # → [Float64]. Always.

# ╔═╡ 49546ce1-11ee-4cc8-b412-24b9e933a728
md"""
**Q3 (board / discussion).** The error on π decreases like $1/\sqrt{n}$. To gain one decimal digit you thus need ~100× more points. **Why is it the *perfect* candidate for parallelism?** (Hint: each draw is completely independent of the others — we say *embarrassingly parallel*. Keep this idea for the threads module.)
"""

# ╔═╡ b7792bef-9262-42b4-98d0-407113ded36c
# Q3: free space to experiment (vary n and observe the error)
let
    for n in (10^4, 10^5, 10^6, 10^7, 10^8)
        err = abs(estimate_pi(n) - π)
        println(@sprintf("n = 10^%d   error = %.5f", round(Int, log10(n)), err))
    end
end

# ╔═╡ 765124ae-9c5d-436c-ac54-d3cb26587b87
md"""
## 9. Wrap-up

| Idea | What we saw |
|---|---|
| **Interpreted (Python)** | bytecode + interpretation loop → cost per operation |
| **JIT-compiled (Julia)** | type specialization → native code, no per-operation overhead |
| **numpy / Numba** | fast because they *leave* the Python interpreter (compiled C / restricted JIT) |
| **Type stability** | the condition for Julia to be fast; `@code_warntype`, no globals |
| **Benchmark** | `@btime`/`@benchmark`, interpolate variables |

### Takeaway

> Speed doesn't come from the language "in itself", but from **what the machine must do per operation**. Compiling and specializing removes the overhead; once memory is on the table, only one enemy remains: **the time to fetch data from memory**.

### What's next

First Julia's **multiple dispatch** (vs Python's object-orientation); then **threads** and a classic case: **the parallel sum that gives a wrong result**.
"""

# ╔═╡ 5973d547-5a91-4d4b-8696-88dfbf03a0a3
md"""
## 10. Bonus — watch π converge (interactive)

Each dart is a random point in the unit square. **Green** landed inside the quarter disk (x² + y² ≤ 1), **red** outside. The estimate is **4 × (green / n)**, and it closes in on π as the darts pile up.

Drag the slider and watch it settle — slowly, like 1/√n (that's Q3, seen live).
"""

# ╔═╡ c687db86-e0a1-4083-ba96-d5a9af91738c
md"""**Number of darts:** $(@bind n_pts Slider(100:100:3000, default=800, show_value=true))"""

# ╔═╡ 7f18d64c-dc08-4d31-850a-a7e3bfc4bdcf
let
    S = 360
    rng = MersenneTwister(1)
    pts = [(rand(rng), rand(rng)) for _ in 1:n_pts]
    inside(p) = p[1]^2 + p[2]^2 <= 1.0
    nin = count(inside, pts)
    est = 4 * nin / n_pts
    Xp(x) = round(x * S, digits=1)
    Yp(y) = round(S * (1 - y), digits=1)
    arc = join(("$(Xp(cos(t))),$(Yp(sin(t)))" for t in range(0, pi/2, length=80)), " ")
    dots = join(("<circle cx='$(Xp(p[1]))' cy='$(Yp(p[2]))' r='2.5' fill='$(inside(p) ? "#2a9d8f" : "#e76f51")' fill-opacity='0.75'/>" for p in pts))
    HTML("""
    <div style="font-family:sans-serif">
      <svg width="$S" height="$S" viewBox="0 0 $S $S" style="background:#fff;border:1px solid #bbb">
        <polygon points="$(Xp(0)),$(Yp(0)) $arc" fill="#a8dadc" fill-opacity="0.3"/>
        <polyline points="$arc" fill="none" stroke="#1d3557" stroke-width="2.5"/>
        <rect x="0.5" y="0.5" width="$(S-1)" height="$(S-1)" fill="none" stroke="#333" stroke-width="1.5"/>
        $dots
      </svg>
      <div style="margin-top:6px;font-size:17px">
        <b>π ≈ $(round(est, digits=4))</b> · n = $n_pts · inside = $nin · error = $(round(abs(est - pi), digits=4))
      </div>
    </div>
    """)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "db49b271b4149145d21642645bfe62ddc84408ba"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "468dbe2b510c876dc091b2c74ed52c7c34f48b9b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"
"""

# ╔═╡ Cell order:
# ╟─ab181e11-6516-4169-9189-fb4488f4c8d3
# ╟─e4c6629b-29ce-4e63-b529-04bf1020ee4f
# ╠═86593a2f-41c2-464d-905a-70b368785d81
# ╠═7525bf04-f5c7-45aa-b826-642c15b68eaf
# ╠═d4572c88-919a-4a54-9581-8f241edc1d7a
# ╟─9c279c3b-f2c3-4a98-8f21-b21190182e5b
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81176
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81177
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81178
# ╠═26289681-0477-4670-950f-8351be0fdd5a
# ╠═da96898d-efc5-4e47-a73e-8e0a991bc0f8
# ╟─bb0ace77-07a4-4504-80c1-edc793f3f899
# ╟─99808d58-8875-42ed-bd94-c7e06ab36c10
# ╠═a8128f4d-9cdc-4cd2-8025-33c5935757e0
# ╟─05891168-bf4c-42c6-86f1-5d038606e267
# ╠═b1180cde-dd1e-4983-aae0-bd631034c0df
# ╟─97acf5d7-5b90-4b3a-a2b9-ffdda2d75a52
# ╟─22d3648b-a71a-4299-8e1b-46f032a74046
# ╠═a51ea6fa-9f15-453f-bcea-852c7e71a354
# ╟─e8d202f2-8080-4e46-aa29-157c5a707c13
# ╠═26118708-e6fa-4cdb-a5e9-7a72a376b84f
# ╟─c3c88228-8ec2-4a08-9384-7da546be08fb
# ╠═c14f0bba-a2a2-457d-a400-35224ec23b34
# ╠═eb5c0a66-0e2e-4eba-9cc6-0b6f6acd4fb6
# ╟─3e4c87e0-e1aa-4593-8f71-a57b44ef14fe
# ╠═cf7f8128-23b8-4c2e-a122-d2705f2c43c6
# ╠═d5000000-0000-4d00-9b00-00000000d001
# ╟─cb125025-5a35-4e36-bf13-7efc3e29abd4
# ╟─656f12ed-0f30-498d-8462-f58bff519eda
# ╠═6f3bc084-05bc-4b17-98f6-5d306b03740e
# ╠═c3c32bc0-9d67-46e2-bdbe-c7d3f84ee2f8
# ╠═42fbb9f9-240e-487c-8f3d-ad839115f426
# ╟─e8b6b134-eb4f-49f2-8f39-eb3c3f1365e9
# ╠═9301ea0f-f270-4015-8acd-f23507b82804
# ╟─ba010064-49f9-4872-a4fe-a42d2e754d75
# ╠═9301ea0f-f270-4015-8acd-f23507b82805
# ╟─9301ea0f-f270-4015-8acd-f23507b82806
# ╠═9301ea0f-f270-4015-8acd-f23507b82807
# ╟─ba010064-49f9-4872-a4fe-a42d2e754d76
# ╠═8e0cdaff-257b-4836-8998-2cd5f0ef8044
# ╠═0662912c-e7ab-4223-82d2-e2683543f99b
# ╟─c62ea649-7bc8-4b64-b662-b874c41547f5
# ╟─fad9827d-f3db-43ae-b0bb-b58aca083e79
# ╠═fa9596a7-23bd-4675-998b-af232a7b08e8
# ╠═fa9596a7-23bd-4675-998b-af232a7b08e9
# ╠═41165e5b-f988-4698-97d1-04ca43eb7964
# ╟─761ce2e0-503b-48b1-9a5d-2b2cce7f4f45
# ╠═ba6f002b-32d3-4327-a14e-ad049956e1ec
# ╟─8d4f26b4-4d42-4291-8c31-0891ba765a37
# ╠═0fcbe125-0551-42b2-93b7-2fe0f6e65bb2
# ╟─bd7de916-7afd-4b63-81e3-2f82fc00b935
# ╠═f53445b9-d198-4ad4-9414-140afdb6460c
# ╠═20c2f0e8-b528-40cf-94b5-e308ae298137
# ╟─46d41ce1-700e-41eb-8ae4-02a37802725b
# ╟─260d7cdf-31b5-47d5-8ea5-ccc76e270dbe
# ╟─d02cd083-dea1-417f-a941-8eac616197a6
# ╟─46c146f1-d1e4-4797-87bc-2c03a0185daf
# ╠═ca49af93-0baa-47ee-a4ec-3849a1cacffb
# ╟─d1000000-0000-4a00-8000-000000000001
# ╠═d1000000-0000-4a00-8000-000000000002
# ╠═d1000000-0000-4a00-8000-000000000003
# ╠═d1000000-0000-4a00-8000-000000000004
# ╟─d1000000-0000-4a00-8000-000000000005
# ╠═d1000000-0000-4a00-8000-000000000006
# ╠═d1000000-0000-4a00-8000-000000000007
# ╟─d1000000-0000-4a00-8000-000000000008
# ╠═d1000000-0000-4a00-8000-000000000009
# ╠═d1000000-0000-4a00-8000-00000000000a
# ╟─d1000000-0000-4a00-8000-00000000000b
# ╟─49ea0717-5e4a-44df-a895-11b9d55c747f
# ╠═760cd379-4178-4e7f-9695-a1d709b36969
# ╠═8adcfe41-1d0d-475a-968c-5a5a971a142b
# ╟─855d77ed-ca4b-407a-b225-2be1a2c499f5
# ╠═3cb6763c-dc00-4c09-94b5-289f773dc460
# ╟─f15bd89d-47f6-4c87-a3d9-17078b722f1e
# ╠═654b39b4-b592-4066-b519-0783a704f637
# ╠═654b39b4-b592-4066-b519-0783a704f638
# ╟─49546ce1-11ee-4cc8-b412-24b9e933a728
# ╠═b7792bef-9262-42b4-98d0-407113ded36c
# ╟─765124ae-9c5d-436c-ac54-d3cb26587b87
# ╟─5973d547-5a91-4d4b-8696-88dfbf03a0a3
# ╟─c687db86-e0a1-4083-ba96-d5a9af91738c
# ╟─7f18d64c-dc08-4d31-850a-a7e3bfc4bdcf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
