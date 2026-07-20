### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ aa07d25d-4f02-4013-95a0-49f72ddf7323
md"""
# Parallel computing — Memory and GPU
## The memory hierarchy, batching, and the GPU

**ENSAI 3A**

---

So far: *compiling and specializing* removes the **per-operation** overhead; *parallelizing* demands that we **minimize sharing** (otherwise: wrong results).

One last enemy remains — the most important one in practice:

> **The processor spends most of its time *waiting* for memory.**

Today:

1. The **memory hierarchy** (registers → L1 → L2 → L3 → RAM) and its orders of magnitude.
2. **Locality**: why the *same* computation is 2 to 10× slower depending on the access order (and the Julia *column-major* vs numpy *row-major* trap).
3. The ML message: **batching turns a memory-bound problem into a compute-bound one**.
4. The **GPU**, where this principle becomes vital — with Monte-Carlo π ported to the graphics card.
"""

# ╔═╡ ee0a05ee-935e-42c3-b567-5624fb902747
md"""
## 0. Getting started
"""

# ╔═╡ 3b62bc4c-9765-4424-b7ee-2800c085103a
begin
    using BenchmarkTools
    using PlutoUI
    using Printf
    using LinearAlgebra
    using CUDA          # with no GPU the package still loads;
                        # CUDA.functional() is what tells us whether it's usable
    md"Packages loaded ✓"
end

# ╔═╡ e60ad78b-4669-47b1-b819-443e4454f585
md"""
## 1. The memory hierarchy — the orders of magnitude at stake

A CPU core can do a floating-point addition in **less than a nanosecond**. But fetching a piece of data from RAM takes **~100 nanoseconds**. In that time the core could have done **hundreds** of operations. If it waits, it does nothing.

To soften this, the hardware inserts **caches** — smaller and faster the closer they get — between the core and RAM:

| Level | Typical size | Approx. latency | Analogy (if L1 = 1 s) |
|---|---|---|---|
| Registers | ~a few hundred bytes | ~0 | instant |
| **L1** cache | ~32–64 KB | ~1 ns | **1 second** |
| **L2** cache | ~256 KB–1 MB | ~4 ns | 4 seconds |
| **L3** cache | ~8–32 MB | ~15 ns | 15 seconds |
| **RAM** | ~16–64 GB | ~100 ns | **~2 minutes** |
| SSD | TB | ~100 µs | **~1 day** |

The idea to remember: **the bigger it is, the further away, the slower.** A cache works on a bet — **locality**:

- **temporal locality**: if we just used a value, we'll probably reuse it soon → keep it close.
- **spatial locality**: if we use a value, we'll probably use **its neighbours** → so the cache loads memory in **lines** (~64 bytes, i.e. 8 `Float64`) at a time.

The whole art of fast code is **honouring that bet**.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c001
md"""
### Seeing the steps: the "cache cliff" experiment

That table can be **measured**. We sum an array of growing size, sweeping it several times (so it stays "hot" in the smallest cache that holds it), and look at the **time per access**:

- as long as the array **fits in L1**, each access is nearly free;
- as soon as it **spills** into L2, then L3, then RAM, the time per access climbs **in steps** — you *see* the hierarchy appear.

⚠️ The thresholds depend on **your** machine: look for the **jumps**, not the absolute values.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c002
# Sum `a` a large number of times: the memory footprint stays that of `a`, but we
# multiply the accesses → we measure the speed of the cache level that holds `a`.
function repeated_sum(a, passes)
    s = zero(eltype(a))
    for _ in 1:passes
        @inbounds @simd for i in eachindex(a)
            s += a[i]
        end
    end
    return s
end

# ╔═╡ cb000000-0000-4c00-9a00-00000000c003
let
    println("footprint | time per access (ns)")
    println("-"^32)
    for kb in (8, 24, 64, 256, 1024, 4096, 16384, 65536)
        n = kb * 128                       # KB → number of Float64 (1024/8 = 128)
        a = rand(n)
        passes = max(1, 100_000_000 ÷ n)   # ≈ constant total access volume
        t = @belapsed repeated_sum($a, $passes)
        ns = t / (passes * n) * 1e9
        footprint = kb < 1024 ? @sprintf("%4d KB", kb) : @sprintf("%4d MB", kb ÷ 1024)
        println(@sprintf("%9s | %14.3f", footprint, ns))
    end
end

# ╔═╡ 58cf2a8a-433f-4737-8116-044f7f8eada1
md"""
## 2. Locality, measured: walking a matrix the right way

This is **the** central lab of this module. We sum every element of a matrix two ways:

- walking it **column by column**;
- walking it **row by row**.

It is **exactly the same computation**, the same number of additions. Yet the times differ widely. Why? Because of how the matrix is **laid out in memory**.

> **Crucial point — Julia stores matrices by *columns* (column-major).** Two elements of the same *column* are neighbours in memory. This is **the opposite of C / numpy**, which store by *rows* (row-major). If you carry your numpy reflexes over unchanged, you'll be going against the grain every time.
"""

# ╔═╡ 3218651a-5b26-42d1-b330-917d0801ed44
begin
    const A = rand(8_000, 8_000)   # 64M Float64 ≈ 512 MB: far beyond the cache
    size(A), Base.summarysize(A) ÷ 1_000_000, "MB"
end

# ╔═╡ 2126f361-6be9-4f14-a358-6a034213fa4f
# COLUMN by column: the row index (i) varies fastest.
# In column-major, A[i,j] and A[i+1,j] are NEIGHBOURS in memory → spatial locality honoured.
function sum_by_columns(M)
    s = zero(eltype(M))
    @inbounds for j in axes(M, 2)        # for each column
        for i in axes(M, 1)              # walk down the column
            s += M[i, j]
        end
    end
    return s
end

# ╔═╡ 6908e575-a27a-43b8-b7ee-2cf0bce7f8fc
# ROW by row: the column index (j) varies fastest.
# A[i,j] and A[i,j+1] are a whole column apart in memory → we jump all over the place.
function sum_by_rows(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)        # for each row
        for j in axes(M, 2)             # walk across the row
            s += M[i, j]
        end
    end
    return s
end

# ╔═╡ 43f403f1-a036-4454-8670-327d079dd5cc
sum_by_columns(A) ≈ sum_by_rows(A)       # same mathematical result

# ╔═╡ cdad600b-7259-4388-a83b-0a2f42f83aa1
let
    print("columns (with the grain) : "); @btime sum_by_columns($A)
    print("rows    (against it)     : "); @btime sum_by_rows($A)
end

# ╔═╡ 62969c4a-d889-4322-8923-725b34f839dc
md"""
### What we just saw

The **same computation**, merely reordered, is several times slower.

- **`sum_by_columns`**: we read memory in the order it is stored. Every cache line loaded (64 bytes = 8 `Float64`) is **fully used** before moving to the next. The spatial-locality bet is won.
- **`sum_by_rows`**: at each step we jump a whole column's length. We load a cache line to use **a single element** of it, then throw it away. We pay the memory-access cost far more often.

> **Julia rule**: the loop over the **first index** (the rows) must be the **inner** loop. That's the opposite of C/numpy. Getting it wrong doesn't give a wrong result — just slow code, which is sneakier.
"""

# ╔═╡ 09844b2e-6544-494d-a7c1-50c3d5b9c5d6
md"""
**Quick activity.** Vary the size below with the slider. For a **small** matrix that fits in cache, the gap disappears (everything is already "close"). The gap only appears once the data **spills out of the cache**. That's direct proof it really is a cache effect.
"""

# ╔═╡ 842ce51a-2b62-49e4-a447-1b842c6fe632
@bind matsize Slider(100:100:4000, default=2000, show_value=true)

# ╔═╡ 01d52dcd-e327-418c-9906-dddd778b01f0
let
    B = rand(matsize, matsize)
    tc = @belapsed sum_by_columns($B)
    tr = @belapsed sum_by_rows($B)
    kbytes = Base.summarysize(B) ÷ 1024
    @sprintf("matrix %d×%d (%d KB) — columns: %.2f ms | rows: %.2f ms | ratio: %.1f×",
             matsize, matsize, kbytes, tc*1e3, tr*1e3, tr/tc)
end

# ╔═╡ ad57f7bf-d863-4be8-b5d4-d0dfd0603b69
md"""
## 3. From locality to batching: the key ML message

Here we get to the heart of it. Let's introduce a notion: **arithmetic intensity** = the number of compute operations performed **per byte loaded** from memory.

- **Low** intensity → we spend our time loading data while the compute units wait: we are **memory-bound**.
- **High** intensity → each loaded value is reused a lot, the compute units run flat out: we are **compute-bound**.

Let's compare two linear-algebra operations.
"""

# ╔═╡ 05884f27-4254-4a8e-8177-e38454dc8aac
begin
    const n = 2000
    const Mat = rand(n, n)
    const xvec = rand(n)
    const Mat2 = rand(n, n)
    md"Data ready."
end

# ╔═╡ dc815f82-3a2e-40a7-b9a1-67daf6c38766
let
    # matrix-VECTOR product: each coefficient of Mat is read ONCE, barely reused.
    # ~2n² operations for ~n² values read → intensity ~constant: MEMORY-BOUND.
    print("matrix × vector : "); @btime $Mat * $xvec

    # matrix-MATRIX product: each coefficient is reused n times.
    # ~2n³ operations for ~n² values → intensity ∝ n: COMPUTE-BOUND.
    print("matrix × matrix : "); @btime $Mat * $Mat2
end

# ╔═╡ fc30c6c4-6c52-4bad-b28e-5f54f21b4b1c
md"""
### Estimating the orders of magnitude

Count it **per byte** — not per *value* (8× apart for `Float64`). For `n = 2000`, `W` weighs `8·n²` = **32 MB**:

- **Matrix × vector**: ~2·n² ≈ 8 MFLOP out of those 32 MB → **0.25 FLOP per byte**, and it stays 0.25 whatever `n` is. Once a coefficient is loaded we barely use it: the computation waits on memory.
- **Matrix × matrix**: ~2·n³ ≈ 16 GFLOP out of the **same** 32 MB → **n/4 = 500 FLOP per byte**, growing with `n`. Each coefficient is reused ~n times (that's the *tiling* / blocking BLAS performs) → the compute units saturate.

Same bytes loaded, **n = 2000× more arithmetic extracted from them**. That's why matrix-matrix (BLAS level 3, `GEMM`) reaches nearly the machine's **peak performance**, while matrix-vector (level 2) tops out at the **memory bandwidth**.

!!! warning "Don't compare their seconds"
    `matrix × matrix` is far *slower* above — because it does n× more **work**. Seconds
    can't compare them. The honest unit is **GFLOP/s** (work per second): measured here,
    ~13 GFLOP/s for `mat×vec` vs ~240 for `mat×mat`. One number, and it tells you by
    itself which resource is saturated.

### The link with machine learning

A neural network is essentially matrix products.

- **Infer on 1 single example** → matrix-**vector** products → *memory-bound* → the hardware is underused.
- **Infer on a *batch* of B examples** → the B vectors stack into a **matrix** → matrix-**matrix** products → *compute-bound* → the hardware works flat out.

> **"We batch to go faster"** is not a magic recipe: it is **turning a memory-bound problem into a compute-bound one**. The same work per example, but reusing each loaded weight for every example in the batch instead of just one.

And this is *even more* true on a GPU — where the underuse penalty is enormous. That's what the end of this module is about.
"""

# ╔═╡ ba7c1100-0001-4a01-8b01-aaaaaaaa0001
md"""
### Seeing it directly: throughput **per example**

⚠️ The benchmark above compared *one* matrix-vector to *one* matrix-matrix. But they don't do the same amount of work: the matrix-matrix just looks "longer". That does **not** demonstrate the point of batching.

The real test is: **for a given example, how much time PER EXAMPLE?** We take a "layer" `W` (n inputs → n outputs) and apply it to a *batch* of `B` examples stacked as columns (`W * X`, with `X` of size `n×B`). We look at the time per **single** example as `B` grows.
"""

# ╔═╡ ba7c1100-0002-4a01-8b01-aaaaaaaa0002
let
    nb = 4096                        # W = 4096² Float32 = 64 MiB — well past the 24 MiB L3
    W = rand(Float32, nb, nb)        # one dense layer: nb → nb
    println("batch B | total (ms) | µs PER EXAMPLE | GFLOP/s")
    println("-"^52)
    for B in (1, 8, 64, 256, 1024)
        X = rand(Float32, nb, B)             # B examples stacked as columns
        t = @belapsed $W * $X seconds=1      # ONE matrix product for the whole batch
        println(@sprintf("%6d  | %10.2f | %14.2f | %7.1f",
                         B, t*1e3, t/B*1e6, 2.0*nb*nb*B/t/1e9))
    end
end

# ╔═╡ ba7c1100-0003-4a01-8b01-aaaaaaaa0003
md"""
The **total** time grows with `B` (we compute more), but the time **per example** *drops*: one and the same weight matrix `W`, loaded from memory, is reused for all `B` examples in the batch instead of just one. This is exactly "moving from matrix-vector (memory-bound) to matrix-matrix (compute-bound)" — and *that*, concretely, is where batching's speedup comes from.

Measured on this machine (cold — see the warning): **2551.8 µs/example at `B=1` → 138.5 at `B=1024` = 18.4× better per example**, and **13.1 → 242.3 GFLOP/s**: 18.4× more of the *same* CPU actually used. **Batching is not a GPU trick** — it already pays here, before any card is involved.

!!! warning "Two honesty notes"
    **1.** At `B = 1`, `W * X` with `X` of size `(nb, 1)` is a **skinny GEMM** (N=1), about
    **2.3× slower** than a true `gemv` (`W * x`: 2.55 ms vs 1.09 ms). Against that fairer
    baseline the gain is **~7.9×**, not 18.4×. Same lesson, no strawman.

    **2.** This is a **laptop**: the CPU and the GPU share a power budget. Benchmark the CPU
    right after hammering the GPU and it throttles — measured **75 ms vs 2.97 ms for the
    same gemm**. Run this cell *before* the GPU sections, or on a cooled machine.
"""

# ╔═╡ 92cd196c-2cab-4aee-aecb-89f06133486c
md"""
## 4. The GPU: thousands of threads, provided you feed them

A CPU is a few very fast, very "clever" cores. A GPU is **thousands** of simpler cores, designed to run **the same operation on different data** en masse (the **SIMT** model: *Single Instruction, Multiple Threads*).

Two consequences:

- **Enormous memory bandwidth**, and a memory latency **hidden by parallelism**: when thousands of threads wait on memory, the GPU switches to other ready threads. So you need **a lot** of simultaneous work to "fill" the machine (*occupancy*).
- **A small problem badly underuses the GPU.** For a mere 100 elements, the CPU↔GPU transfer and the launch cost dominate the computation itself. **You need big batches.**

Same conclusion as section 3, pushed to the extreme: a GPU is only worth it if you feed it **massive**, regular parallel work.

> ⚙️ **The following cells need an NVIDIA GPU + `CUDA.jl`.** If you don't have one, read them: they are commented so they stay understandable, and a "CPU fallback" version is provided further down.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c004
md"""
### CPU caches vs GPU caches: two opposite strategies

The GPU **also** has a memory hierarchy — the *principle* is identical (memory is far, so bring the data closer):

| Level | CPU | GPU (NVIDIA) |
|---|---|---|
| closest | registers | registers (per thread) |
| scratchpad / L1 | L1 cache | **shared memory** + L1 (per SM) |
| shared | L2, L3 caches | L2 cache (across SMs) |
| far | RAM (DDR) | global memory (VRAM/HBM) |

But the **strategy** is the opposite:

- **How do you hide latency?** The CPU bets on **big caches** (megabytes per core) for *few* threads. The GPU bets on **a very large number of threads**: when a *warp* waits on memory, the hardware switches to another ready warp → you need **many** threads to hide the wait (*occupancy*). That's precisely what "give the GPU enough work" means.
- **Who manages it?** CPU caches are **automatic** (transparent). On a GPU, *shared memory* is a **scratchpad explicitly managed by the program**: you decide what to put there (a "manual" L1).
- **Spatial locality.** On the CPU: read **neighbouring** slots (64-byte cache lines) — that's the column/row walk of section 2. On the GPU, the analogue is **coalescing**: the 32 threads of a *warp* must read **contiguous** addresses so the hardware can merge their accesses into a single transaction.

> **Same problem, two strategies.** CPU: *big caches, few threads*. GPU: *small caches, a huge number of threads + an explicit scratchpad*. **Locality** is decisive on both sides — we measure it on the GPU right after.
"""

# ╔═╡ c3790242-0bd4-4168-858a-3e58d5bb3134
# Is the GPU actually usable? (card + driver + CUDA all working)
gpu_available = CUDA.functional()

# ╔═╡ e366e8e3-9f59-40bd-bb00-89347808fb1f
if gpu_available
    let
        CUDA.versioninfo()
    end
else
    md"⚠️ **No CUDA GPU detected.** The GPU cells will print a message; section 6 provides a CPU fallback."
end

# ╔═╡ cb000000-0000-4c00-9a00-00000000c005
md"""
### On the GPU side: coalescing, measured

The GPU analogue of "walking the right way". We read the **same** array through an index array:

- **contiguous** (`idx[i] = i`): neighbouring threads read neighbouring addresses → the hardware **merges** their reads into as few transactions as possible (32 consecutive `Float32` = 128 B = **4 sectors of 32 B**) = *coalesced* access;
- **large stride** (`idx[i] = stride·i`): neighbouring threads read **far** from one another → up to 32 separate accesses = **non-coalesced**.

Same number of reads, same result — only the **access order** changes. This is exactly the spatial locality of section 2, transposed to the *warp*.

!!! warning "Why the stride is 33 and not 32"
	The index has to be a real **permutation**, or we measure something else entirely. `(i·s) % N` lands on multiples of `gcd(s, N)`, so it covers only `N / gcd(s, N)` distinct slots. With `N = 20_000_000` and `s = 32`: `gcd = 32` → just **625 000 distinct indices**, each read 32× — a 2.4 MiB working set that **fits in L2**. That would be a cache demo wearing a coalescing costume, and "same number of reads" would be false.

	`gcd(33, 20_000_000) = 1` → a true permutation over all 20 M slots. One character, and the sentence above becomes true.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c006
if gpu_available
    let
        N = 20_000_000
        a = CUDA.rand(Float32, N)
        contiguous = CuArray(Int32.(1:N))                                # neighbours → coalesced
        stride_len = 33                                                  # gcd(33, N) = 1 → permutation
        scattered  = CuArray(Int32.(((0:N-1) .* stride_len) .% N .+ 1))  # big stride → non-coalesced
        print("contiguous access (coalesced) : "); @btime CUDA.@sync $a[$contiguous]
        print("strided access (scattered)    : "); @btime CUDA.@sync $a[$scattered]
    end
else
    md"*(GPU demo skipped — no card detected)*"
end

# ╔═╡ cb000000-0000-4c00-9a00-00000000c007
md"""
The strided access is clearly slower **for the same number of reads**: the threads of a warp no longer land on the same line, so the GPU must issue far more memory transactions. It is the exact counterpart of the "against the grain" walk from section 2 on the CPU — **locality** decides the speed on both sides.

*"Only ~2.4×? I thought coalescing was decisive."* — Do the traffic accounting. `out = a[idx]` reads the **index** array (80 MB) and writes the **result** (80 MB), both perfectly coalesced whatever the stride, and gathers 80 MB from `a`. So **two thirds of the traffic is coalesced by construction**, and only one third can ever be penalized — the ratio is capped no matter how bad the gather gets. The lesson isn't "2.4× is small": it's that you can only speed up **the part you actually control**. That's Amdahl's law again, this time on memory traffic.
"""

# ╔═╡ bf891e84-9d69-454a-a77b-724596266414
md"""
### Demonstration: the matrix product, CPU vs GPU

We move the matrices onto the GPU with `cu(...)`, and launch the **same** `*`. Multiple dispatch (seen earlier!) automatically picks the GPU version because the arguments are now `CuArray`. Same source code, different hardware.
"""

# ╔═╡ 78901ba1-a4f3-4808-99ed-b151bab95b78
if gpu_available
    let
        N = 4096
        Acpu = rand(Float32, N, N)
        Bcpu = rand(Float32, N, N)
        Agpu = cu(Acpu)              # copy to the GPU's memory
        Bgpu = cu(Bcpu)

        print("CPU (Float32) : ")
        @btime $Acpu * $Bcpu

        print("GPU (Float32) : ")
        # CUDA.@sync waits for the GPU computation to finish (it is asynchronous!)
        @btime CUDA.@sync $Agpu * $Bgpu
    end
else
    md"*(GPU demo skipped — no card detected)*"
end

# ╔═╡ a3b77b02-29a2-455e-823d-d6068cfe3199
md"""
On a large matrix the gap is big (often **10 to 50×**). But on a **small** problem the GPU can be **slower** than the CPU, because of transfer and launch costs. **A GPU only pays off at scale** — which is what the next activity is about.
"""

# ╔═╡ ba7c1100-0004-4a01-8b01-aaaaaaaa0004
md"""
### Batching on the GPU: giving it enough work

Let's redo the **throughput per example** experiment (section 3), but comparing **CPU and GPU** as the batch size grows. A layer `W` (m → m) applied to `B` examples stacked as columns (`W * X`, `X` of size `m×B`).

The GPU has thousands of cores: at `B = 1` (a plain matrix-vector) most of them sit idle. The bigger the batch, the higher the utilisation.
"""

# ╔═╡ ba7c1100-0005-4a01-8b01-aaaaaaaa0005
if gpu_available
    let
        m = 4096                              # dense layer m → m, in Float32
        Wc = rand(Float32, m, m); Wg = cu(Wc)
        println("batch B | CPU µs/example | GPU µs/example | GPU speedup")
        println("-"^58)
        for B in (1, 8, 64, 256, 1024)
            Xc = rand(Float32, m, B); Xg = cu(Xc)
            tc = @belapsed $Wc * $Xc                  # CPU
            tg = @belapsed CUDA.@sync $Wg * $Xg       # GPU (asynchronous → @sync)
            println(@sprintf("%6d  | %13.3f | %14.3f | %10.1f×",
                             B, tc/B*1e6, tg/B*1e6, tc/tg))
        end
    end
else
    md"*(GPU demo skipped — no card detected; the CPU version is in section 3)*"
end

# ╔═╡ ba7c1100-0006-4a01-8b01-aaaaaaaa0006
md"""
On the GPU the time **per example** drops sharply as `B` grows: at `B = 1` the GPU is massively underused (often **slower** than the CPU), while at large `B` it clearly beats it. This is the most direct illustration that you must give the GPU enough work — and the reason why, in production, inference is always done in **batches**.
"""

# ╔═╡ 1ed9c688-150c-423c-a086-181603547e2c
md"""
## 5. Activity — when does the GPU start paying off?

We'll plot the GPU/CPU speedup **as a function of problem size**, to see the tipping point with our own eyes.
"""

# ╔═╡ 36c37805-567f-4baf-b8d3-5d42d5027b88
if gpu_available
    let
        println("size N |  CPU (ms) |  GPU (ms) |  speedup")
        println("-"^48)
        for N in (64, 128, 256, 512, 1024, 2048, 4096)
            Ac = rand(Float32, N, N); Bc = rand(Float32, N, N)
            Ag = cu(Ac);              Bg = cu(Bc)
            tc = @belapsed $Ac * $Bc
            tg = @belapsed CUDA.@sync $Ag * $Bg
            println(@sprintf("%7d  | %8.3f  | %8.3f  |  %6.1f×", N, tc*1e3, tg*1e3, tc/tg))
        end
    end
else
    md"*(GPU activity skipped — no card detected)*"
end

# ╔═╡ 7663fb47-1f0c-4383-a5dd-e55f9a1b5abc
md"""
**Q1.** From which size `N` on does the GPU become faster than the CPU? For small sizes, why is it *slower*? (Hint: memory transfer + *kernel* launch, fixed costs independent of N.)

**Q2.** Does the speedup keep growing with N, or does it plateau? What does that ceiling tell you about the machine (think "bandwidth" vs "peak compute")?
"""

# ╔═╡ e85db6c9-b322-4a2f-aed6-295f21cb362d
md"""
## 6. The red thread: Monte-Carlo π on the GPU

We carried this computation from sequential to multithreaded CPU. Last step: the GPU.

Monte-Carlo is **ideal** for a GPU: millions of completely independent draws. But remember **Q4 of the threads module** — the one where `rand()` turned out *not* to be shared state: on the CPU, each **task** carries its own `TaskLocalRNG` stream, which is why `pi_spawn` had no race and needed no lock.

The same requirement now has to hold for **millions of GPU threads at once**, not a couple of dozen tasks: **the generator must give every thread an independent stream** — cheaply, and without any of them talking to each other. `CUDA.jl` provides exactly that (`CUDA.rand` draws on the device, one stream per thread). Same design problem as Q4, three orders of magnitude up.

Rather than writing an explicit *kernel*, we express the computation in a **vectorised** way: we draw all the points directly on the GPU and count those inside the quarter disk — all the data stays on the card.
"""

# ╔═╡ 7a6d400a-f826-4b67-9a7d-717351daa9bf
# "Vectorised" GPU version: no explicit loop, CUDA parallelises everything.
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # n draws, straight into GPU memory
    y = CUDA.rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0      # Bool vector, computed on the GPU
    return 4 * sum(inside) / n           # sum reduces on the GPU
end

# ╔═╡ aa59c041-e5fa-4cb0-a8f0-43c353255569
# Equivalent CPU fallback, for comparison (and for those without a GPU)
function pi_cpu_vec(n)
    x = rand(Float32, n)
    y = rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0
    return 4 * sum(inside) / n
end

# ╔═╡ 798cfc8b-51b7-4086-8b30-faa6be29c299
if gpu_available
    let
        n = 100_000_000
        println("π (reference) : ", π)
        println("pi_gpu : ", pi_gpu(n))
        print("CPU vectorised : "); @btime pi_cpu_vec($n)
        print("GPU            : "); @btime CUDA.@sync pi_gpu($n)
    end
else
    let
        n = 20_000_000
        println("π (reference) : ", π)
        print("CPU vectorised : "); @btime pi_cpu_vec($n)
        println("(no GPU: the pi_gpu version needs CUDA)")
    end
end

# ╔═╡ 0d3be671-416b-4c2f-a605-1995a602e132
md"""
**Q3.** With 100 million draws, measure the GPU/CPU speedup. Compare it as well with the gain we got from **CPU threads**. Does the GPU give you much more than the number of CPU cores? Why (how many "threads" does a GPU launch, in orders of magnitude)?

**Q4 (course synthesis).** Go back to Amdahl's law. In `pi_gpu`, what **stays sequential** or costly and doesn't benefit from massive parallelism? (Hint: allocating the vectors, the final `sum` reduction, and — if you're not careful — the CPU↔GPU transfers.) That's what stops the speedup from being "infinite".
"""

# ╔═╡ 130f0c20-5ca5-4045-9e70-e3898789bf01
md"""
## 7. Wrap-up — and course synthesis

### What we saw today

| Idea | Message |
|---|---|
| **Memory hierarchy** | the CPU *waits* for memory; caches bet on locality |
| **Column-major (Julia)** | inner loop on the 1st index; the opposite of numpy |
| **Arithmetic intensity** | few ops/byte = *memory-bound*; many = *compute-bound* |
| **Batching** | stack the examples: matrix-vector → matrix-matrix = memory-bound → compute-bound |
| **GPU** | thousands of threads; only worth it at scale, and only if fed enough |

### The synthesis, one sentence per module

1. **Compiling and specializing** removes the *per-operation* overhead (interpreted Python → compiled Julia).
2. **Multiple dispatch** makes functions both generic *and* fast — and it's what flips a `*` onto the GPU without changing the code.
3. **Parallelizing** demands *minimizing sharing*, otherwise the result is wrong (the parallel sum).
4. **Feeding the hardware**: organise the data to honour the cache, and batch, until the GPU saturates.

### The takeaway

> Performance doesn't come from the language itself. It comes from **understanding what the machine actually does**: the cost of each operation, the data that is shared, and how it moves through memory. Julia lets you observe all of it — from lowered code down to assembly, from a concurrency bug up to the GPU. These are principles, not recipes: they hold in any language.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─aa07d25d-4f02-4013-95a0-49f72ddf7323
# ╟─ee0a05ee-935e-42c3-b567-5624fb902747
# ╠═3b62bc4c-9765-4424-b7ee-2800c085103a
# ╟─e60ad78b-4669-47b1-b819-443e4454f585
# ╟─cb000000-0000-4c00-9a00-00000000c001
# ╠═cb000000-0000-4c00-9a00-00000000c002
# ╠═cb000000-0000-4c00-9a00-00000000c003
# ╟─58cf2a8a-433f-4737-8116-044f7f8eada1
# ╠═3218651a-5b26-42d1-b330-917d0801ed44
# ╠═2126f361-6be9-4f14-a358-6a034213fa4f
# ╠═6908e575-a27a-43b8-b7ee-2cf0bce7f8fc
# ╠═43f403f1-a036-4454-8670-327d079dd5cc
# ╠═cdad600b-7259-4388-a83b-0a2f42f83aa1
# ╟─62969c4a-d889-4322-8923-725b34f839dc
# ╟─09844b2e-6544-494d-a7c1-50c3d5b9c5d6
# ╠═842ce51a-2b62-49e4-a447-1b842c6fe632
# ╠═01d52dcd-e327-418c-9906-dddd778b01f0
# ╟─ad57f7bf-d863-4be8-b5d4-d0dfd0603b69
# ╠═05884f27-4254-4a8e-8177-e38454dc8aac
# ╠═dc815f82-3a2e-40a7-b9a1-67daf6c38766
# ╟─fc30c6c4-6c52-4bad-b28e-5f54f21b4b1c
# ╟─ba7c1100-0001-4a01-8b01-aaaaaaaa0001
# ╠═ba7c1100-0002-4a01-8b01-aaaaaaaa0002
# ╟─ba7c1100-0003-4a01-8b01-aaaaaaaa0003
# ╟─92cd196c-2cab-4aee-aecb-89f06133486c
# ╟─cb000000-0000-4c00-9a00-00000000c004
# ╠═c3790242-0bd4-4168-858a-3e58d5bb3134
# ╠═e366e8e3-9f59-40bd-bb00-89347808fb1f
# ╟─cb000000-0000-4c00-9a00-00000000c005
# ╠═cb000000-0000-4c00-9a00-00000000c006
# ╟─cb000000-0000-4c00-9a00-00000000c007
# ╟─bf891e84-9d69-454a-a77b-724596266414
# ╠═78901ba1-a4f3-4808-99ed-b151bab95b78
# ╟─a3b77b02-29a2-455e-823d-d6068cfe3199
# ╟─ba7c1100-0004-4a01-8b01-aaaaaaaa0004
# ╠═ba7c1100-0005-4a01-8b01-aaaaaaaa0005
# ╟─ba7c1100-0006-4a01-8b01-aaaaaaaa0006
# ╟─1ed9c688-150c-423c-a086-181603547e2c
# ╠═36c37805-567f-4baf-b8d3-5d42d5027b88
# ╟─7663fb47-1f0c-4383-a5dd-e55f9a1b5abc
# ╟─e85db6c9-b322-4a2f-aed6-295f21cb362d
# ╠═7a6d400a-f826-4b67-9a7d-717351daa9bf
# ╠═aa59c041-e5fa-4cb0-a8f0-43c353255569
# ╠═798cfc8b-51b7-4086-8b30-faa6be29c299
# ╟─0d3be671-416b-4c2f-a605-1995a602e132
# ╟─130f0c20-5ca5-4045-9e70-e3898789bf01
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
