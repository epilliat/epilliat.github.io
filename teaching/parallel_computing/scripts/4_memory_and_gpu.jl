# =============================================================================
# Memory and GPU — the memory hierarchy, batching, and the GPU
# ENSAI 3A — Julia as the test bench, Python as the point of comparison
# -----------------------------------------------------------------------------
# ⚠ KEEP IN SYNC with `pluto/4_memory_and_gpu.jl` — same lesson, two formats.
# Cells are delimited by `#%%` (Alt+Enter runs a cell in VS Code).
# Needs: BenchmarkTools, CUDA (the GPU sections are skipped without a card).
#
# The story so far: compiling removes the per-operation overhead; parallelising
# demands minimising sharing. One enemy left, the most important in practice:
#
#     THE PROCESSOR SPENDS MOST OF ITS TIME *WAITING* FOR MEMORY.
#
# 1. the hierarchy · 2. locality · 3. batching · 4. the GPU (and the bus to it).
# =============================================================================

#%% 0. Getting started
using BenchmarkTools
using LinearAlgebra
using CUDA                 # loads without a card; CUDA.functional() tells us
println("CUDA functional: ", CUDA.functional())

#%% 1. The memory hierarchy
#   Level        Typical size      Latency    Analogy (if L1 = 1 s)
#   registers    a few hundred B   ~0         instant
#   L1           32–64 KB          ~1 ns      1 second
#   L2           256 KB–1 MB       ~4 ns      4 seconds
#   L3           8–32 MB           ~15 ns     15 seconds
#   RAM          16–64 GB          ~100 ns    ~2 minutes
#   SSD          TB                ~100 µs    ~1 day
#
# A cache bets on LOCALITY: temporal (reuse soon) and spatial (use the neighbours —
# memory moves in LINES of 64 B = 8 Float64). Fast code honours that bet.

#%% The "cache cliff" — sum the same array many times, grow its footprint
function repeated_sum(a, passes)
    s = zero(eltype(a))
    for _ in 1:passes
        @inbounds @simd for i in eachindex(a)
            s += a[i]
        end
    end
    return s
end

#%% Measure it — look for the JUMPS, not the absolute values
let
    println("footprint | time per access (ns)")
    println("-"^32)
    for kb in (8, 24, 64, 256, 1024, 4096, 16384, 65536)
        n = kb * 128                       # KB → number of Float64
        a = rand(n)
        passes = max(1, 100_000_000 ÷ n)   # ≈ constant total access volume
        t = @belapsed repeated_sum($a, $passes)
        ns = t / (passes * n) * 1e9
        footprint = kb < 1024 ? "$(lpad(kb, 4)) KB" : "$(lpad(kb ÷ 1024, 4)) MB"
        println("$(lpad(footprint, 9)) | $(lpad(round(ns, digits=3), 14))")
    end
end
# The steps are real — but the cliff is ~6.5×, not the 100× the table promises.
# The table is LATENCY (one isolated access); this loop is a predictable stream, so
# the PREFETCHER hides the latency and we pay BANDWIDTH instead. Random access
# (pointer chasing) has nothing to prefetch and does pay the full ~100 ns.

#%% 2. Locality — the same maths, walked two ways
# ⚠ Julia is COLUMN-major: A[i,j] and A[i+1,j] are neighbours. numpy/C is the
#   opposite. Bring your numpy reflexes over unchanged and you go against the grain.
A = rand(8_000, 8_000)   # 64M Float64 ≈ 512 MB: far beyond any cache
println(size(A), " — ", Base.summarysize(A) ÷ 1_000_000, " MB")

#%% The two walks
function sum_by_columns(M)          # row index varies fastest → with the grain
    s = zero(eltype(M))
    @inbounds for j in axes(M, 2)       # for each column
        for i in axes(M, 1)             # walk down the column
            s += M[i, j]
        end
    end
    return s
end

function sum_by_rows(M)             # column index varies fastest → against it
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)       # for each row
        for j in axes(M, 2)             # walk across the row
            s += M[i, j]
        end
    end
    return s
end

@assert sum_by_columns(A) ≈ sum_by_rows(A)   # same mathematical result

#%% Time them
let
    print("columns (with the grain) : "); @btime sum_by_columns($A)
    print("rows    (against it)     : "); @btime sum_by_rows($A)
end
# With the grain, every 64 B line loaded is fully used. Against it, we load a line
# to use ONE element and throw it away. RULE: the loop over the FIRST index goes
# INNERMOST. Getting it wrong isn't a wrong answer, just slow code — sneakier.

#%% Proof it IS the cache: shrink the matrix until it fits
let
    println("  size  |  columns (ms) |  rows (ms) | ratio")
    println("-"^46)
    for m in (100, 500, 1000, 2000, 4000)
        B = rand(m, m)
        tc = @belapsed sum_by_columns($B)
        tr = @belapsed sum_by_rows($B)
        kb = Base.summarysize(B) ÷ 1024
        println("$(lpad(m, 5))² ($(lpad(kb, 6)) KB) | $(lpad(round(tc*1e3, digits=2), 8)) | " *
                "$(lpad(round(tr*1e3, digits=2), 8)) | $(lpad(round(tr/tc, digits=1), 5))x")
    end
end
# Small matrix → no gap at all. The gap appears exactly when the data outgrows the
# cache. So the claim is not "columns are fast", it is "the cache decides".

#%% 3. Arithmetic intensity — FLOP per byte loaded
#   low  → the compute units wait for loads  → MEMORY-BOUND
#   high → each loaded value is reused a lot → COMPUTE-BOUND
n = 2000
Mat = rand(n, n)
xvec = rand(n)
Mat2 = rand(n, n)

let
    print("matrix × vector : "); @btime $Mat * $xvec    # 2n² FLOP / 8n² bytes
    print("matrix × matrix : "); @btime $Mat * $Mat2    # 2n³ FLOP / 8n² bytes
end
# Per byte: mat×vec = 0.25 FLOP/byte, CONSTANT in n. mat×mat = n/4, GROWS with n.
# Same bytes loaded, 2000× more arithmetic extracted from them.
# ⚠ Never compare their SECONDS — they do different amounts of work. Compare
#   GFLOP/s (here ~13 vs ~240): one number, and it says which resource is saturated.

#%% Batching: same layer W, B examples stacked as columns of X
let
    nb = 4096                        # W = 4096² Float32 = 64 MiB — past the 24 MiB L3
    W = rand(Float32, nb, nb)
    println("batch B | total (ms) | µs PER EXAMPLE | GFLOP/s")
    println("-"^52)
    for B in (1, 8, 64, 256, 1024)
        X = rand(Float32, nb, B)
        t = @belapsed $W * $X seconds=1      # ONE matrix product for the whole batch
        println("$(lpad(B, 6))  | $(lpad(round(t*1e3, digits=2), 10)) | " *
                "$(lpad(round(t/B*1e6, digits=2), 14)) | $(lpad(round(2.0*nb*nb*B/t/1e9, digits=1), 7))")
    end
end
# Total time grows (more work), time PER EXAMPLE collapses: 2551.8 → 138.5 µs = 18.4×,
# 13.1 → 242.3 GFLOP/s. W is streamed from RAM once per batch — a FIXED cost divided
# by B — so the curve flattens once that cost is amortised. Flattening = success.
# BATCHING IS NOT A GPU TRICK: it converts memory-bound into compute-bound, right here
# on the CPU. (A neural net is matrix products: 1 example = mat×vec, a batch = mat×mat.)
# ⚠ Two honesty notes: B=1 here is a skinny GEMM, ~2.3× slower than a true gemv, so
#   the fair gain is ~7.9×; and on a laptop the CPU throttles if you benchmark it
#   right after the GPU (75 ms vs 2.97 ms for the same gemm). Run this cell first.

#%% 4. The GPU — thousands of slow threads instead of a few fast ones
#   CPU: few fast cores, BIG automatic caches, hides latency by CACHING.
#   GPU: thousands of simple cores (SIMT), small per-thread caches + an explicit
#        scratchpad (shared memory), hides latency by SWITCHING to another warp.
#   → the GPU needs an OCEAN of identical work to fill; a small problem underuses it.
gpu_available = CUDA.functional()
gpu_available && CUDA.versioninfo()

#%% A GPU array is NOT a CPU array — different type, different memory
# `cu(x)` COPIES the data over the PCIe bus into the card's VRAM and hands back a
# CuArray. It is a different object living in a different address space.
if gpu_available
    let
        M  = rand(3, 3)                 # Matrix{Float64}, in RAM
        Mg = cu(M)                      # CuArray{Float32,...}, in VRAM
        println("CPU : ", typeof(M))
        println("GPU : ", typeof(Mg))
        println("⚠ cu() also converts Float64 → Float32: ", eltype(M), " → ", eltype(Mg))
        println("  (CuArray(M) keeps the eltype: ", eltype(CuArray(M)), ")")
        # Reading ONE element back means a round trip over the bus — CUDA.jl refuses:
        r = try Mg[1] catch e; e end
        println("Mg[1] → ", r isa Exception ? "ERROR (scalar indexing disallowed)" : r)
        println("  get it all back with Array(Mg), or reduce on the card first.")
    end
end
# 🐍 Same split in PyTorch (python/4): `torch.rand` is float32 where numpy is float64,
#    `.cuda()` copies, `t.device` says which side you are on, and mixing the two raises
#    "Expected all tensors to be on the same device" — the classic beginner error.

#%% What the copy costs
if gpu_available
    let
        N = 4096
        Ac = rand(Float32, N, N); Bc = rand(Float32, N, N)   # 64 MiB each
        Ag = cu(Ac); Bg = cu(Bc)
        t_up   = @belapsed cu($Ac)                    # host → device
        t_down = @belapsed Array($Ag)                 # device → host
        t_mul  = @belapsed CUDA.@sync($Ag * $Bg)      # the actual work
        println("host → device : $(round(t_up*1e3, digits=2)) ms")
        println("device → host : $(round(t_down*1e3, digits=2)) ms")
        println("matmul on GPU : $(round(t_mul*1e3, digits=2)) ms")
        println("→ the two transfers cost $(round((t_up + t_down)/t_mul, digits=1))× the computation")
    end
end
# Measured here: 8.5 ms up, 51 ms down, for a ~35 ms matmul — the round trip costs
# MORE than the computation it feeds, and coming BACK is the expensive direction.
# RULE: upload ONCE, do many operations on the card, bring back only what you need
# (ideally a scalar). Code that ping-pongs after every step lives on the bus, not on
# the GPU. That is why pi_gpu below draws its random numbers ON the card.

#%% Work IN PLACE — a fresh output is memory traffic you chose to pay
# `z = 2 .* x .+ y` allocates 76 MB every call. `z .= ...` writes into memory you
# already own. Note the dot on the `=` — that is the whole difference.
let
    n = 10_000_000
    x = rand(n); y = rand(n); z = zeros(n)
    print("allocates : "); @btime 2 .* $x .+ $y
    print("in place  : "); @btime $z .= 2 .* $x .+ $y
end
# ~3× here, and 76 MiB → 0 allocations. In a loop that runs 20 times: 2.6 s and
# 1.56 GiB of garbage vs 0.43 s and nothing. Same rule for BLAS — `A * B` allocates
# the result, `mul!(C, A, B)` reuses C (measured 195 ms → 79 ms at n = 2000):
let
    n = 2000
    Ai = rand(n, n); Bi = rand(n, n); Ci = zeros(n, n)
    print("C = A * B     : "); @btime $Ai * $Bi
    print("mul!(C, A, B) : "); @btime mul!($Ci, $Ai, $Bi)
end
# ⚠ Measure, don't assume: in place is NOT automatically faster. For matrix-VECTOR
#   the gain vanishes (the output is tiny), and on the GPU it vanishes too — CUDA.jl
#   allocates from a pool, so there is no RAM to stream. On the card the cost that
#   matters is the one above: the TRANSFER.

#%% The matrix product, CPU vs GPU — the SAME `*`
# The arguments are CuArrays now, so MULTIPLE DISPATCH (the language module!) picks
# the GPU method. Not one line of the algorithm changed.
if gpu_available
    let
        N = 4096
        Acpu = rand(Float32, N, N); Bcpu = rand(Float32, N, N)
        Agpu = cu(Acpu);            Bgpu = cu(Bcpu)
        print("CPU (Float32) : "); @btime $Acpu * $Bcpu
        print("GPU (Float32) : "); @btime CUDA.@sync $Agpu * $Bgpu   # @sync: it is async!
    end
else
    println("(GPU demo skipped — no card detected)")
end

#%% Coalescing — locality again, one bus further out
# The 32 threads of a warp issue their loads TOGETHER. Neighbouring addresses get
# merged into a few 32 B sectors; scattered ones cost one transaction each.
# ⚠ The index must be a real PERMUTATION: (i*s) % N covers only N/gcd(s,N) slots, so
#   s = 32 would be a cache demo, not a coalescing demo. gcd = 1 for 1, 33 and 1023.
if gpu_available
    let
        N = 20_000_000
        a = CUDA.rand(Float32, N)
        println("stride | gcd(s,N) |    time (ms)")
        println("-"^34)
        for s in (1, 33, 1023)
            idx = CuArray(Int32.(((0:N-1) .* s) .% N .+ 1))
            t = @belapsed CUDA.@sync($a[$idx])
            println("$(lpad(s, 6)) | $(lpad(gcd(s, N), 8)) | $(lpad(round(t*1e3, digits=3), 12))")
        end
    end
else
    println("(GPU demo skipped — no card detected)")
end
# "Only ~2.5×?" Do the traffic accounting: `a[idx]` also READS the index array (80 MB)
# and WRITES the result (80 MB), both coalesced whatever the stride. Only 1/3 of the
# traffic can be punished, so the ratio is capped. Amdahl again, on memory traffic.

#%% Testing that explanation — a story you could LOSE
# The accounting makes a PREDICTION: drop the index array (compute the index in the
# kernel) and only 1/2 the traffic stays coalesced, so the ratio MUST rise.
function gather_kernel!(out, a, s, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        j = ((i - 1) * s) % n + 1        # index COMPUTED, never loaded
        @inbounds out[i] = a[j]
    end
    return nothing
end

gather_run!(out, a, s, n) =
    CUDA.@sync @cuda threads=256 blocks=cld(n, 256) gather_kernel!(out, a, s, n)

if gpu_available
    let
        N = 20_000_000
        a = CUDA.rand(Float32, N); out = CUDA.zeros(Float32, N)
        t1  = @belapsed gather_run!($out, $a, 1, $N)
        t33 = @belapsed gather_run!($out, $a, 33, $N)
        println("kernel stride 1 : $(round(t1*1e3, digits=3)) ms")
        println("kernel stride 33: $(round(t33*1e3, digits=3)) ms")
        println("ratio: $(round(t33/t1, digits=2))×  — higher than the gather above?")
    end
end
# Measured: ~2.7× for the gather (2/3 coalesced) vs ~4.2× for the kernel (1/2).
# The prediction held — and it could have failed. That is what makes it an
# explanation rather than an excuse.

#%% Batching on the GPU — read the TOTAL column
if gpu_available
    let
        m = 4096
        Wc = rand(Float32, m, m); Wg = cu(Wc)
        println("batch B | GPU total (ms) | GPU µs/example | GPU GFLOP/s | CPU µs/ex | CPU/GPU")
        println("-"^76)
        for B in (1, 8, 64, 256, 1024)
            Xc = rand(Float32, m, B); Xg = cu(Xc)
            tc = @belapsed $Wc * $Xc
            tg = @belapsed CUDA.@sync $Wg * $Xg
            println("$(lpad(B, 6))  | $(lpad(round(tg*1e3, digits=3), 14)) | " *
                    "$(lpad(round(tg/B*1e6, digits=3), 14)) | " *
                    "$(lpad(round(2.0*m*m*B/tg/1e9, digits=0), 11)) | " *
                    "$(lpad(round(tc/B*1e6, digits=1), 9)) | $(round(tc/tg, digits=1))×")
        end
    end
else
    println("(GPU demo skipped — the CPU version is in section 3)")
end
# The GPU TOTAL barely moves (~0.41 → ~0.45 ms) while B goes 1 → 64: 64× the work for
# 10% more time, because at B = 1 the card was idle. ~1.7% of its GFLOP/s at B = 1,
# ~100% at B = 64.
# ⚠ Don't say "the GPU loses at B = 1" — here it still beats the CPU ~6×. Say "you are
#   using 1.7% of what you paid for": that invites batching instead of giving up.
#   (python/4 measures the same table in PyTorch and gets the same shape: total 0.40 → 0.46 ms
#   from B = 1 to 64, and 84 → 7252 GFLOP/s. Compare the two in class.)

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
            println("$(lpad(N, 7))  | $(lpad(round(tc*1e3, digits=3), 8))  | " *
                    "$(lpad(round(tg*1e3, digits=3), 8))  |  $(lpad(round(tc/tg, digits=1), 6))×")
        end
    end
else
    println("(GPU activity skipped — no card detected)")
end
# Q1. From which N does the GPU win? Why is it SLOWER below that?
#     (Hint: transfer + kernel launch are fixed costs, independent of N.)
# Q2. Does the speedup keep growing, or plateau? What does the ceiling say about the
#     machine — bandwidth, or peak compute?

#%% 6. The red thread: Monte-Carlo π on the GPU
# Everything stays on the card: draw there, compare there, reduce there. Only the
# final scalar crosses the bus.
# Remember Q4 of the threads module — rand() was not shared state because each TASK
# has its own stream. The same must now hold for MILLIONS of GPU threads; CUDA.rand
# gives exactly that. Same design problem, three orders of magnitude up.
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # drawn straight into VRAM
    y = CUDA.rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0
    return 4 * sum(inside) / n           # reduced on the GPU
end

pi_cpu_vec(n) = (x = rand(Float32, n); y = rand(Float32, n);
                 4 * sum((x.^2 .+ y.^2) .<= 1f0) / n)

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
        print("CPU vectorised : "); @btime pi_cpu_vec($n)
        println("(no GPU: pi_gpu needs CUDA)")
    end
end
# Q3. Compare the GPU speedup with the gain from CPU threads. Why so much bigger?
# Q4. Amdahl, again: in pi_gpu, what does NOT benefit from massive parallelism?
#     (Allocating the vectors, the final reduction, and any careless transfer.)

#%% 7. Wrap-up — and course synthesis
# Memory hierarchy     the CPU WAITS for memory; caches bet on locality
# Column-major         inner loop on the 1st index; the opposite of numpy
# Arithmetic intensity few FLOP/byte = memory-bound; many = compute-bound
# Batching             stack examples: mat×vec → mat×mat = memory- → compute-bound
# CPU vs GPU arrays    a CuArray is a DIFFERENT object in a DIFFERENT memory;
#                      transfers are expensive → move once, work in place, reduce there
# GPU                  thousands of threads; worth it only at scale, and only if fed
#
# ONE SENTENCE PER MODULE:
#  1. Compiling and specializing removes the PER-OPERATION overhead.
#  2. Multiple dispatch makes code generic AND fast — and flips a `*` onto the GPU.
#  3. Parallelizing demands MINIMIZING SHARING, else the result is wrong.
#  4. Feed the hardware: honour the cache, batch, and stop moving data around.
#
# TAKEAWAY: performance doesn't come from the language. It comes from what the machine
# actually does — cost per operation, what is shared, and how data moves. These are
# principles, not recipes: they hold in any language.
