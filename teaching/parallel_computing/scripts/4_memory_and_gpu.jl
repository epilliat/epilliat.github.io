# =============================================================================
# Memory and GPU — the memory hierarchy, batching, and the GPU
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC: this is the plain-script twin of `pluto/4_memory_and_gpu.jl`.
#   The Pluto notebook and this script hold the SAME lesson in two formats — edit
#   BOTH together whenever the content changes, so they never drift apart.
# -----------------------------------------------------------------------------
# Cells are delimited by `#%%` (Alt+Enter runs a cell in VS Code).
# Needs: BenchmarkTools, CUDA (the GPU sections are skipped without a card).
#
# So far: compiling and specializing removes the PER-OPERATION overhead;
# parallelizing demands MINIMIZING SHARING (else: wrong results). One last enemy
# remains, the most important in practice:
#
#     THE PROCESSOR SPENDS MOST OF ITS TIME *WAITING* FOR MEMORY.
#
# Today: 1. the memory hierarchy · 2. locality (and the Julia column-major vs numpy
# row-major trap) · 3. batching turns memory-bound into compute-bound · 4. the GPU.
# =============================================================================

#%% 0. Getting started
using BenchmarkTools
using Printf
using LinearAlgebra
using CUDA          # with no GPU the package still loads;
                    # CUDA.functional() is what tells us whether it's usable
println("CUDA functional: ", CUDA.functional())

#%% 1. The memory hierarchy — the orders of magnitude at stake
# A core does a float addition in < 1 ns. Fetching from RAM takes ~100 ns — in that
# time the core could have done HUNDREDS of operations. Caches sit in between:
#
#   Level        Typical size      Latency    Analogy (if L1 = 1 s)
#   registers    a few hundred B   ~0         instant
#   L1           32–64 KB          ~1 ns      1 second
#   L2           256 KB–1 MB       ~4 ns      4 seconds
#   L3           8–32 MB           ~15 ns     15 seconds
#   RAM          16–64 GB          ~100 ns    ~2 minutes
#   SSD          TB                ~100 µs    ~1 day
#
# Bigger = further = slower. A cache bets on LOCALITY:
#   - temporal: just used it → will reuse it soon → keep it close;
#   - spatial : used it → will use its NEIGHBOURS → load whole LINES (~64 B = 8 Float64).
# The whole art of fast code is honouring that bet.

#%% Seeing the steps: the "cache cliff" experiment
# Sum `a` many times: the footprint stays that of `a`, but we multiply the accesses
# → we measure the speed of the cache level that holds `a`.
function repeated_sum(a, passes)
    s = zero(eltype(a))
    for _ in 1:passes
        @inbounds @simd for i in eachindex(a)
            s += a[i]
        end
    end
    return s
end

#%% Measure it — look for the JUMPS, not the absolute values (they are machine-specific)
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
# As long as the array fits in L1 each access is nearly free; once it spills into L2,
# L3, then RAM, the time per access climbs IN STEPS — you SEE the hierarchy appear.

#%% 2. Locality, measured: walking a matrix the right way
# Same computation, same number of additions, two orders. The times differ widely —
# because of how the matrix is LAID OUT in memory.
#
# ⚠ CRUCIAL: Julia stores matrices by COLUMNS (column-major). Two elements of the
#   same column are neighbours in memory. This is THE OPPOSITE of C / numpy
#   (row-major). Carry your numpy reflexes over unchanged and you go against the
#   grain every time.
const A = rand(8_000, 8_000)   # 64M Float64 ≈ 512 MB: far beyond the cache
println(size(A), " — ", Base.summarysize(A) ÷ 1_000_000, " MB")

#%% The two walks
# COLUMN by column: the row index (i) varies fastest.
# In column-major, A[i,j] and A[i+1,j] are NEIGHBOURS → spatial locality honoured.
function sum_by_columns(M)
    s = zero(eltype(M))
    @inbounds for j in axes(M, 2)        # for each column
        for i in axes(M, 1)              # walk down the column
            s += M[i, j]
        end
    end
    return s
end

# ROW by row: the column index (j) varies fastest.
# A[i,j] and A[i,j+1] are a whole column apart → we jump all over the place.
function sum_by_rows(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)        # for each row
        for j in axes(M, 2)             # walk across the row
            s += M[i, j]
        end
    end
    return s
end

@assert sum_by_columns(A) ≈ sum_by_rows(A)   # same mathematical result

#%% Time them — same maths, different speed
let
    print("columns (with the grain) : "); @btime sum_by_columns($A)
    print("rows    (against it)     : "); @btime sum_by_rows($A)
end
# - sum_by_columns: every 64-byte cache line loaded is FULLY used before moving on.
# - sum_by_rows   : we load a cache line to use ONE element, then throw it away.
#
# JULIA RULE: the loop over the FIRST index (rows) must be the INNER loop — the
# opposite of C/numpy. Getting it wrong isn't a wrong result, just slow code, which
# is sneakier.

#%% Proof it's a cache effect: shrink the matrix until it fits
# For a SMALL matrix that fits in cache, the gap disappears (everything is already
# "close"). The gap only appears once the data spills OUT of the cache.
let
    println("  size  |  columns (ms) |  rows (ms) | ratio")
    println("-"^46)
    for m in (100, 500, 1000, 2000, 4000)
        B = rand(m, m)
        tc = @belapsed sum_by_columns($B)
        tr = @belapsed sum_by_rows($B)
        kb = Base.summarysize(B) ÷ 1024
        println(@sprintf("%5d² (%6d KB) | %8.2f | %8.2f | %5.1fx", m, kb, tc*1e3, tr*1e3, tr/tc))
    end
end

#%% 3. From locality to batching: the key ML message
# ARITHMETIC INTENSITY = compute operations performed PER BYTE loaded from memory.
#   low  → we spend our time loading, the compute units wait  → MEMORY-BOUND
#   high → each loaded value is reused a lot, units run flat out → COMPUTE-BOUND
const n = 2000
const Mat = rand(n, n)
const xvec = rand(n)
const Mat2 = rand(n, n)

let
    # matrix-VECTOR: each coefficient of Mat is read ONCE, barely reused.
    # ~2n² ops for ~n² values read → intensity ~constant: MEMORY-BOUND.
    print("matrix × vector : "); @btime $Mat * $xvec
    # matrix-MATRIX: each coefficient is reused n times.
    # ~2n³ ops for ~n² values → intensity ∝ n: COMPUTE-BOUND.
    print("matrix × matrix : "); @btime $Mat * $Mat2
end
# Count it PER BYTE (not per value — 8x apart for Float64). For n = 2000, W is
# 8n² = 32 MB:
#   mat×vec : 2n² = 8 MFLOP  out of that 32 MB → 0.25 FLOP/byte, CONSTANT in n
#   mat×mat : 2n³ = 16 GFLOP out of the SAME 32 MB → n/4 = 500 FLOP/byte, ∝ n
# Same bytes loaded, n = 2000x more arithmetic extracted from them. BLAS-3 tiling is
# what realises it: each tile of W is reused from cache instead of re-streamed. That
# is why GEMM (BLAS-3) approaches PEAK while mat×vec (BLAS-2) tops out at the MEMORY
# BANDWIDTH.
# ⚠ Do NOT compare their SECONDS: mat×mat is far slower simply because it does n×
#   more work. Compare GFLOP/s — that one number says memory-bound vs compute-bound
#   by itself (measured here: ~13 GFLOP/s for mat×vec vs ~240 for mat×mat).

#%% Seeing it directly: throughput PER EXAMPLE
# ⚠ The benchmark above compared ONE mat×vec to ONE mat×mat — they don't do the same
#   amount of work, so it does NOT demonstrate batching. The real test: at a given
#   example count, how much time PER EXAMPLE?
# A "layer" W (n → n) applied to B examples stacked as columns (W * X, X is n×B).
let
    nb = 4096                        # W = 4096² Float32 = 64 MiB — well past the 24 MiB L3
    W = rand(Float32, nb, nb)
    println("batch B | total (ms) | µs PER EXAMPLE | GFLOP/s")
    println("-"^52)
    for B in (1, 8, 64, 256, 1024)
        X = rand(Float32, nb, B)             # B examples stacked as columns
        t = @belapsed $W * $X seconds=1      # ONE matrix product for the whole batch
        println(@sprintf("%6d  | %10.2f | %14.2f | %7.1f",
                         B, t*1e3, t/B*1e6, 2.0*nb*nb*B/t/1e9))
    end
end
# Measured here (cold machine — see the warning below): 2551.8 µs/example at B=1 down
# to 138.5 at B=1024 = 18.4x better per example, and 13.1 → 242.3 GFLOP/s: 18.5x more
# of the SAME CPU actually used. Batching is not a GPU trick — it pays right here.
# ⚠ Two honesty notes.
#   1. At B=1, `W * X` with X of size (nb,1) is a SKINNY GEMM (N=1), ~2.3x slower than
#      a true gemv (`W * x`: 2.55 ms vs 1.09 ms). Against that fairer baseline the gain
#      is ~7.9x, not 18.4x. Still the same lesson, without the strawman.
#   2. This is a LAPTOP: CPU and GPU share a power budget. Benchmark the CPU right
#      after hammering the GPU and it throttles (measured: 75 ms vs 2.97 ms for the
#      SAME gemm). Run this cell before the GPU sections, or on a cooled machine.
# Total time grows with B (we compute more), but time PER EXAMPLE DROPS: the same W,
# loaded once, is reused for all B examples instead of one. That is exactly
# "matrix-vector (memory-bound) → matrix-matrix (compute-bound)".
#
# THE ML LINK: a neural net is essentially matrix products. Infer on 1 example →
# mat×vec → memory-bound → hardware underused. Infer on a BATCH of B → the vectors
# stack into a matrix → mat×mat → compute-bound → hardware flat out.
# "We batch to go faster" is not magic: it is turning a memory-bound problem into a
# compute-bound one.

#%% 4. The GPU: thousands of threads, provided you feed them
# A CPU = a few very fast, "clever" cores. A GPU = THOUSANDS of simpler cores running
# the same operation on different data (SIMT: Single Instruction, Multiple Threads).
#   - Enormous bandwidth, latency HIDDEN BY PARALLELISM: when thousands of threads
#     wait on memory, the GPU switches to other ready threads → you need a LOT of
#     simultaneous work to fill the machine (occupancy).
#   - A small problem badly underuses the GPU: transfer + launch cost dominate.
#
# CPU caches vs GPU caches — same principle, OPPOSITE strategy:
#   closest      | registers        | registers (per thread)
#   scratchpad   | L1 cache         | SHARED MEMORY + L1 (per SM)
#   shared       | L2, L3           | L2 (across SMs)
#   far          | RAM (DDR)        | global memory (VRAM/HBM)
#
#   - Hiding latency: CPU bets on BIG CACHES for FEW threads. GPU bets on a HUGE
#     NUMBER OF THREADS — when a warp waits, the hardware switches warp (occupancy).
#   - Who manages: CPU caches are AUTOMATIC. GPU shared memory is a scratchpad
#     managed EXPLICITLY by the program (a "manual" L1).
#   - Spatial locality: CPU = read neighbouring slots (64 B lines) = section 2. GPU =
#     COALESCING: the 32 threads of a warp must read CONTIGUOUS addresses so the
#     hardware merges them into as few transactions as possible (32 consecutive
#     Float32 = 128 B = 4 sectors of 32 B — not one, but far better than up to 32
#     scattered accesses).
gpu_available = CUDA.functional()
gpu_available && CUDA.versioninfo()

#%% Coalescing, measured (GPU only)
# Same array, read through an index array: contiguous (idx[i]=i) → neighbouring
# threads read neighbouring addresses → the hardware merges them into as FEW
# transactions as possible. Large stride → threads read far apart → many more
# transactions (non-coalesced). Same number of reads, same result — only the
# ACCESS ORDER changes.
#
# ⚠ WHY 33 AND NOT 32: the index must be a true PERMUTATION, or we'd be measuring
# something else entirely. (i*s) % N hits gcd(s, N) values apart, so it covers
# N/gcd(s,N) distinct slots. With N = 20_000_000 and s = 32, gcd = 32 → only
# 625_000 distinct indices, each read 32× — a 2.4 MiB working set that fits in L2.
# That is a cache demo, not a coalescing demo, and "same number of reads" would be
# a lie. gcd(33, 20_000_000) = 1 → a real permutation over all 20 M slots.
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
    println("(GPU demo skipped — no card detected)")
end
# The strided access is clearly slower FOR THE SAME NUMBER OF READS: the warp's
# threads no longer land on one line, so the GPU issues far more transactions. Exact
# counterpart of the "against the grain" walk on CPU — locality decides on both sides.
#
# "Only ~2.4×? You said coalescing was decisive." Do the traffic accounting: out =
# a[idx] reads the INDEX array (80 MB) and writes the RESULT (80 MB) — both perfectly
# coalesced whatever the stride — and gathers 80 MB from `a`. So 2/3 of the traffic is
# coalesced by construction and only 1/3 can be penalized: the ratio is capped no
# matter how bad the gather gets. The lesson isn't "2.4× is small", it's that you can
# only ever speed up the part you actually control. Amdahl again, on memory traffic.

#%% The matrix product, CPU vs GPU
# We move the matrices with cu(...) and launch the SAME `*`. MULTIPLE DISPATCH (the
# language module!) picks the GPU method automatically, because the arguments are now
# CuArray. Same source code, different hardware.
if gpu_available
    let
        N = 4096
        Acpu = rand(Float32, N, N); Bcpu = rand(Float32, N, N)
        Agpu = cu(Acpu);            Bgpu = cu(Bcpu)          # copy to GPU memory
        print("CPU (Float32) : "); @btime $Acpu * $Bcpu
        # CUDA.@sync waits for the GPU computation to finish (it is asynchronous!)
        print("GPU (Float32) : "); @btime CUDA.@sync $Agpu * $Bgpu
    end
else
    println("(GPU demo skipped — no card detected)")
end
# On a large matrix the gap is big (often 10–50x). But on a SMALL problem the GPU can
# be SLOWER than the CPU (transfer + launch costs). A GPU only pays off at scale.

#%% Batching on the GPU: giving it enough work
# The GPU has thousands of cores: at B = 1 (a plain matrix-vector) most sit idle.
if gpu_available
    let
        m = 4096                              # dense layer m → m, Float32
        Wc = rand(Float32, m, m); Wg = cu(Wc)
        println("batch B | CPU µs/example | GPU µs/example | GPU speedup")
        println("-"^58)
        for B in (1, 8, 64, 256, 1024)
            Xc = rand(Float32, m, B); Xg = cu(Xc)
            tc = @belapsed $Wc * $Xc                  # CPU
            tg = @belapsed CUDA.@sync $Wg * $Xg       # GPU (asynchronous → @sync)
            println(@sprintf("%6d  | %13.3f | %14.3f | %10.1f×", B, tc/B*1e6, tg/B*1e6, tc/tg))
        end
    end
else
    println("(GPU demo skipped — no card detected; the CPU version is in section 3)")
end
# Time PER EXAMPLE drops sharply as B grows: at B = 1 the GPU is massively underused
# (often slower than the CPU), at large B it clearly beats it. This is why, in
# production, inference is always done in BATCHES.

#%% 5. Activity — when does the GPU start paying off?
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
    println("(GPU activity skipped — no card detected)")
end
# Q1. From which N does the GPU beat the CPU? For small sizes, why is it SLOWER?
#     (Hint: memory transfer + kernel launch — fixed costs independent of N.)
# Q2. Does the speedup keep growing with N, or plateau? What does that ceiling say
#     about the machine (think "bandwidth" vs "peak compute")?

#%% 6. The red thread: Monte-Carlo π on the GPU
# We carried this from sequential to multithreaded CPU. Last step: the GPU.
# Monte-Carlo is IDEAL for a GPU: millions of independent draws. But remember Q4 of
# the threads module — the one where rand() turned out NOT to be shared state: on the
# CPU each TASK carries its own TaskLocalRNG stream, which is why pi_spawn had no race
# and needed no lock. That same requirement must now hold for MILLIONS of GPU threads
# at once, not a couple of dozen tasks: every thread needs an independent stream,
# cheaply, with none of them talking to each other. CUDA.jl provides exactly that
# (CUDA.rand draws on the device, one stream per thread). Same design problem as Q4,
# three orders of magnitude up.
# Rather than an explicit kernel, we express it VECTORISED: draw every point directly
# on the GPU, count those in the quarter disk — all the data stays on the card.
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # n draws, straight into GPU memory
    y = CUDA.rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0      # Bool vector, computed on the GPU
    return 4 * sum(inside) / n           # sum reduces on the GPU
end

# Equivalent CPU fallback, for comparison (and for those without a GPU)
function pi_cpu_vec(n)
    x = rand(Float32, n)
    y = rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0
    return 4 * sum(inside) / n
end

#%% Measure it
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
# Q3. With 100M draws, measure the GPU/CPU speedup. Compare with the gain we got from
#     CPU THREADS. Does the GPU give much more than the number of CPU cores? Why (how
#     many "threads" does a GPU launch, in orders of magnitude)?
# Q4 (course synthesis). Back to Amdahl. In pi_gpu, what STAYS sequential or costly
#     and doesn't benefit from massive parallelism? (Hint: allocating the vectors, the
#     final `sum` reduction, and — if you're careless — the CPU↔GPU transfers.)

#%% 7. Wrap-up — and course synthesis
# Memory hierarchy    the CPU WAITS for memory; caches bet on locality
# Column-major        inner loop on the 1st index; the opposite of numpy
# Arithmetic intensity few ops/byte = memory-bound; many = compute-bound
# Batching            stack examples: mat×vec → mat×mat = memory-bound → compute-bound
# GPU                 thousands of threads; only worth it at scale, and only if fed
#
# ONE SENTENCE PER MODULE:
#  1. Compiling and specializing removes the PER-OPERATION overhead (Python → Julia).
#  2. Multiple dispatch makes functions generic AND fast — and flips a `*` onto the
#     GPU without changing the code.
#  3. Parallelizing demands MINIMIZING SHARING, else the result is wrong (parallel sum).
#  4. Feed the hardware: organise data to honour the cache, and batch, until the GPU
#     saturates.
#
# TAKEAWAY: performance doesn't come from the language itself. It comes from
# understanding WHAT THE MACHINE ACTUALLY DOES: the cost of each operation, the data
# that is shared, and how it moves through memory. Julia lets you observe all of it —
# from lowered code down to assembly, from a concurrency bug up to the GPU. These are
# principles, not recipes: they hold in any language.
