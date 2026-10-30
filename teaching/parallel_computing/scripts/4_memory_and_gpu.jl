# =============================================================================
# Memory and GPU — the memory hierarchy, batching, and the GPU
# ⚠ KEEP IN SYNC with `pluto/4_memory_and_gpu.jl` — same lesson, two formats.
# Cells: `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools and CUDA.jl installed
# (`Pkg.add("CUDA")`), even without a card: the GPU cells then skip.
# For each cell: predict, then run. Sections 1–3 = memory deck, 4–7 = GPU deck.
# =============================================================================

#%% 0. Getting started
using BenchmarkTools
using LinearAlgebra
using CUDA                 # needs the package; a card is optional
gpu_available = CUDA.functional()
println("CUDA functional: $gpu_available")

#%% 1. The memory hierarchy — the cache cliff
#   L1 ~32–64 KB ~1 ns · L2 ~1 MB ~4 ns · L3 ~8–32 MB ~15 ns · RAM ~100 ns
# Same array summed many times, growing footprint. Where does the curve jump?
# Compare with your cache sizes (`lscpu`). Is the RAM step as big as the table says?
function repeated_sum(a, passes)
    s = zero(eltype(a))
    for _ in 1:passes
        @inbounds @simd for i in eachindex(a)
            s += a[i]
        end
    end
    return s
end

let
    println("footprint | time per access (ns)")
    for kb in (8, 24, 64, 256, 1024, 4096, 16384, 65536)
        n = kb * 128                       # KB → number of Float64
        a = rand(n)
        passes = max(1, 100_000_000 ÷ n)   # ≈ constant total access volume
        t = @belapsed repeated_sum($a, $passes) seconds=1
        ns = t / (passes * n) * 1e9
        footprint = kb < 1024 ? "$(lpad(kb, 4)) KB" : "$(lpad(kb ÷ 1024, 4)) MB"
        println("$(lpad(footprint, 9)) | $(lpad(round(ns, digits=3), 14))")
    end
end

#%% 2. Locality — the same sum, walked two ways
# Julia is COLUMN-major (numpy is row-major). Predict which walk is faster.
A = rand(8_000, 8_000)   # 512 MB: far beyond any cache

function sum_by_columns(M)
    s = zero(eltype(M))
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)             # 1st index innermost
            s += M[i, j]
        end
    end
    return s
end

function sum_by_rows(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)
        for j in axes(M, 2)             # 2nd index innermost
            s += M[i, j]
        end
    end
    return s
end

@assert sum_by_columns(A) ≈ sum_by_rows(A)

let
    print("columns : "); @btime sum_by_columns($A)
    print("rows    : "); @btime sum_by_rows($A)
end

#%% Proof it IS the cache: shrink the matrix. At which size does the gap vanish?
let
    println("  size  |  columns (ms) |  rows (ms) | ratio")
    for m in (100, 500, 1000, 2000, 4000)
        B = rand(m, m)
        tc = @belapsed sum_by_columns($B) seconds=1
        tr = @belapsed sum_by_rows($B) seconds=1
        kb = Base.summarysize(B) ÷ 1024
        println("$(lpad(m, 5))² ($(lpad(kb, 6)) KB) | $(lpad(round(tc*1e3, digits=2), 8)) | " *
                "$(lpad(round(tr*1e3, digits=2), 8)) | $(lpad(round(tr/tc, digits=1), 5))x")
    end
end

#%% Allocating is memory traffic too — note the dot on `=`. Look at time AND allocations.
let
    n = 10_000_000
    x = rand(n); y = rand(n); z = zeros(n)
    print("allocates : "); @btime 2 .* $x .+ $y
    print("in place  : "); @btime $z .= 2 .* $x .+ $y
end
# Same for BLAS: `A * B` allocates, `mul!(C, A, B)` reuses C. Does it always pay?
let
    n = 2000
    Ai = rand(n, n); Bi = rand(n, n); Ci = zeros(n, n)
    print("C = A * B     : "); @btime $Ai * $Bi
    print("mul!(C, A, B) : "); @btime mul!($Ci, $Ai, $Bi)
end

#%% 3. Arithmetic intensity — FLOP per byte loaded
# Both read the same n² matrix. Don't compare the seconds: compute the GFLOP/s
# (GFLOP/s = FLOP / time / 1e9; FLOP = one `+` or `*`, counts in the comments).
n = 2000
Mat = rand(n, n)
xvec = rand(n)
Mat2 = rand(n, n)

let
    print("matrix × vector : "); @btime $Mat * $xvec    # 2n² FLOP
    print("matrix × matrix : "); @btime $Mat * $Mat2    # 2n³ FLOP
end

#%% Batching: one layer W, B examples stacked as columns of X
# Watch µs PER EXAMPLE. Why does it drop, on a CPU? (Run before the GPU cells:
# a laptop CPU throttles after the GPU has been hammered.)
let
    nb = 4096                        # W = 64 MiB Float32: past the L3, comes from RAM
    W = rand(Float32, nb, nb)
    println("batch B | total (ms) | µs PER EXAMPLE | GFLOP/s")
    for B in (1, 8, 64, 256, 1024)
        X = rand(Float32, nb, B)
        t = @belapsed $W * $X seconds=1
        println("$(lpad(B, 6))  | $(lpad(round(t*1e3, digits=2), 10)) | " *
                "$(lpad(round(t/B*1e6, digits=2), 14)) | $(lpad(round(2.0*nb*nb*B/t/1e9, digits=1), 7))")
    end
end

#%% 4. The GPU — a CuArray is NOT a CPU array
# What changed in the type? In the element type? What does `Ag[1]` do?
if gpu_available
    let
        A  = rand(3, 3)
        Ag = cu(A)                      # copies to VRAM (the card's own memory)
        println("CPU : ", typeof(A))
        println("GPU : ", typeof(Ag), "   (CuArray(A) keeps ", eltype(CuArray(A)), ")")
        r = try Ag[1] catch e; e end    # one element = one round trip over the bus
        println("Ag[1] → ", r isa Exception ? "ERROR (scalar indexing disallowed)" : r)
    end
end

#%% What the copy costs — compare the transfers with the computation they feed
if gpu_available
    let
        N = 4096
        Ac = rand(Float32, N, N); Bc = rand(Float32, N, N)
        Ag = cu(Ac); Bg = cu(Bc)
        t_up   = @belapsed cu($Ac) seconds=1
        t_down = @belapsed Array($Ag) seconds=1
        t_mul  = @belapsed CUDA.@sync($Ag * $Bg) seconds=1
        println("host → device : $(round(t_up*1e3, digits=2)) ms")
        println("device → host : $(round(t_down*1e3, digits=2)) ms")
        println("matmul on GPU : $(round(t_mul*1e3, digits=2)) ms")
    end
end

#%% The matrix product, CPU vs GPU — the SAME `*` (multiple dispatch picks the method)
if gpu_available
    let
        N = 4096
        Acpu = rand(Float32, N, N); Bcpu = rand(Float32, N, N)
        Agpu = cu(Acpu);            Bgpu = cu(Bcpu)
        print("CPU : "); @btime $Acpu * $Bcpu
        print("GPU : "); @btime CUDA.@sync $Agpu * $Bgpu   # @sync: GPU calls are async
    end
end

#%% 5. Feed the GPU — when does it start paying off? Guess N before running.
if gpu_available
    let
        println("size N |  CPU (ms) |  GPU (ms) |  speedup")
        for N in (64, 128, 256, 512, 1024, 2048, 4096)
            Ac = rand(Float32, N, N); Bc = rand(Float32, N, N)
            Ag = cu(Ac);              Bg = cu(Bc)
            tc = @belapsed $Ac * $Bc seconds=1
            tg = @belapsed CUDA.@sync($Ag * $Bg) seconds=1
            println("$(lpad(N, 7))  | $(lpad(round(tc*1e3, digits=3), 8))  | " *
                    "$(lpad(round(tg*1e3, digits=3), 8))  |  $(lpad(round(tc/tg, digits=1), 6))×")
        end
    end
end
# Q1. Why is the GPU SLOWER for small N? (Which costs don't depend on N?)
# Q2. Does the speedup keep growing, or plateau? What does the ceiling say?

#%% Batching on the GPU — read the TOTAL column. Is the GPU busy at B = 1?
if gpu_available
    let
        m = 4096
        Wc = rand(Float32, m, m); Wg = cu(Wc)
        println("batch B | GPU total (ms) | GPU GFLOP/s | CPU/GPU")
        for B in (1, 8, 64, 256, 1024)
            Xc = rand(Float32, m, B); Xg = cu(Xc)
            tc = @belapsed $Wc * $Xc seconds=1
            tg = @belapsed CUDA.@sync($Wg * $Xg) seconds=1
            println("$(lpad(B, 6))  | $(lpad(round(tg*1e3, digits=3), 14)) | " *
                    "$(lpad(round(2.0*m*m*B/tg/1e9, digits=0), 11)) | $(round(tc/tg, digits=1))×")
        end
    end
end

#%% 6. Coalescing — locality again, at the scale of a warp (32 threads)
# Same reads, different order. Why must the index be a permutation (gcd(s, N) = 1)?
if gpu_available
    let
        N = 20_000_000
        a = CUDA.rand(Float32, N)
        println("stride | gcd(s,N) |    time (ms)")
        for s in (1, 33, 1023)
            idx = CuArray(Int32.(((0:N-1) .* s) .% N .+ 1))
            t = @belapsed CUDA.@sync($a[$idx]) seconds=1
            println("$(lpad(s, 6)) | $(lpad(gcd(s, N), 8)) | $(lpad(round(t*1e3, digits=3), 12))")
        end
    end
end
# Smaller ratio than you expected? Which other bytes does `a[idx]` read and write?

#%% Test that explanation: compute the index in the kernel (no idx array to read).
# Prediction: the ratio must RISE. Does it?
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
        t1  = @belapsed gather_run!($out, $a, 1, $N) seconds=1
        t33 = @belapsed gather_run!($out, $a, 33, $N) seconds=1
        println("kernel stride 1 / 33 : $(round(t1*1e3, digits=3)) / $(round(t33*1e3, digits=3)) ms" *
                " — ratio $(round(t33/t1, digits=2))×")
    end
end

#%% 7. Monte-Carlo π on the GPU — same count as before, 100 million threads
# Everything stays on the card; only a scalar comes back.
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # drawn straight into VRAM
    y = CUDA.rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0    # 1f0 = 1 as a Float32
    return 4 * sum(inside) / n           # reduced on the GPU
end

pi_cpu_vec(n) = (x = rand(Float32, n); y = rand(Float32, n);
                 4 * sum((x.^2 .+ y.^2) .<= 1f0) / n)   # mirrors pi_gpu, allocates

function count_in_disk(n)                # the scalar loop of the earlier modules
    hits = 0
    for _ in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 ≤ 1
            hits += 1
        end
    end
    return hits
end

let
    n = 100_000_000
    print("CPU scalar loop : "); @btime 4 * count_in_disk($n) / $n
    print("CPU vectorized  : "); @btime pi_cpu_vec($n)
    if gpu_available
        println("pi_gpu = ", pi_gpu(n), "   (π = ", Float64(π), ")")
        print("GPU             : "); @btime CUDA.@sync pi_gpu($n)
    end
end
# Q3. Compare with the gain from CPU threads (threads module: 6.3× on 22 threads). Why so much bigger?
# Q4. Amdahl again: what in pi_gpu does NOT benefit from massive parallelism?

#%% 8. Wrap-up
# Performance doesn't come from the language but from what the machine does:
# cost per operation (compile), what is shared (threads), how data moves (cache,
# batching, the bus to the GPU).
