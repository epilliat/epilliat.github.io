# =============================================================================
# Memory and GPU — the memory hierarchy, batching, and the GPU
# ⚠ KEEP IN SYNC with `pluto/4_memory_and_gpu.jl` — same lesson, two formats.
# Cells: `#%%` (Alt+Enter in VS Code). Needs BenchmarkTools and CUDA.jl installed
# (`Pkg.add("CUDA")`), even without a card: the GPU cells then skip.
# Sections 1–3 (+ two optional cells) = memory deck, 4–7 = GPU deck.
# =============================================================================

#%% 0. Getting started
using BenchmarkTools
using LinearAlgebra
using CUDA                 # needs the package; a card is optional
gpu_available = CUDA.functional()
println("CUDA functional: $gpu_available")

#%% 1. Repeated sums as the array grows
#   L1 ~32–64 KB ~1 ns · L2 ~1 MB ~4 ns · L3 ~8–32 MB ~15 ns · RAM ~100 ns
# Same array summed many times, growing footprint. Predict the average time per element.
# Compare footprints with cache sizes (`lscpu`); this is processing rate, not access latency.
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
    println("footprint | average ns per element")
    for kb in (8, 24, 64, 256, 1024, 4096, 16384, 65536)
        n = kb * 128                       # KB → number of Float64
        a = rand(n)
        passes = max(1, 100_000_000 ÷ n)   # ≈ constant total access volume
        t = @belapsed repeated_sum($a, $passes) seconds=1
        ns = t / (passes * n) * 1e9
        footprint = kb < 1024 ? "$(lpad(kb, 4)) KiB" : "$(lpad(kb ÷ 1024, 4)) MiB"
        println("$(lpad(footprint, 9)) | $(lpad(round(ns, digits=3), 14))")
    end
end

#%% 2. Locality — the same sum, walked two ways
# Same additions, two loop orders.
A = rand(8_000, 8_000)   # 512 MB: larger than the reference laptop’s caches

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

#%% Test the effect of matrix size: how does the loop-order ratio change?
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

#%% 3. Batching: one layer W, B examples stacked as columns of X
# Compare W*x with W*X. Matrix multiplication can reuse weights across examples; measure time per example.
# (Run before the GPU cells: CPU/GPU power and thermal state can affect laptop timings.)
let
    nb = 4096                        # W = 64 MiB, larger than the reference laptop's 24 MiB L3
    W = rand(Float32, nb, nb)
    Xall = rand(Float32, nb, 1024)
    x = Xall[:, 1]
    println("BLAS threads: $(BLAS.get_num_threads()); Float32; W bytes: $(sizeof(W))")
    t_ref = @belapsed $W * $x seconds=1
    println("one by one, W * x : $(round(t_ref*1e6, digits=1)) µs per example")
    println("batch B | ms per batch | µs PER EXAMPLE | vs one by one")
    for B in (1, 8, 64, 256, 1024)
        X = Xall[:, 1:B]  # nested subsets of the same input examples
        Y = W * X
        for j in axes(X, 2)
            @assert isapprox(Y[:, j], W * X[:, j]; rtol=1e-4)
        end
        t = @belapsed $W * $X seconds=1
        println("$(lpad(B, 6))  | $(lpad(round(t*1e3, digits=2), 10)) | " *
                "$(lpad(round(t/B*1e6, digits=2), 14)) | $(lpad(round(t_ref*B/t, digits=1), 6))×")
    end
end

#%% Optional — reuse output buffers: note the dot on `=`. Look at time AND allocations.
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

#%% Optional — arithmetic intensity: FLOP per byte loaded
# Use GFLOP/s to compare computational throughput; elapsed time still matters for latency
# (GFLOP/s = FLOP / time / 1e9; FLOP = one `+` or `*`, counts in the comments).
n = 2000
Mat = rand(n, n)
xvec = rand(n)
Mat2 = rand(n, n)

let
    print("matrix × vector : "); @btime $Mat * $xvec    # 2n² FLOP
    print("matrix × matrix : "); @btime $Mat * $Mat2    # 2n³ FLOP
end

#%% 4. Device and precision
# What changed in the type? In the element type? What does `Ag[1]` do?
if gpu_available
    let
        A = rand(Float32, 3, 3)
        Ag = CuArray(A)  # preserve dtype while copying to device memory
        println("CPU: ", typeof(A), "; GPU: ", typeof(Ag))
        println("cu(Float64 input) uses ", eltype(cu(Float64.(A))), "; CuArray preserves ", eltype(CuArray(Float64.(A))))
        CUDA.allowscalar(false)
        r = try Ag[1] catch e; e end
        println("Host scalar indexing rejected: ", r isa Exception)
    end
end

#%% What the copy costs — compare the transfers with the computation
if gpu_available
    let
        N = 4096
        Ac = rand(Float32, N, N); Bc = rand(Float32, N, N)
        Ag = cu(Ac); Bg = cu(Bc)
        t_up   = @belapsed CUDA.@sync(CuArray($Ac)) seconds=1
        t_down = @belapsed Array($Ag) seconds=1
        t_mul  = @belapsed CUDA.@sync($Ag * $Bg) seconds=1
        println("host → device : $(round(t_up*1e3, digits=2)) ms")
        println("device → host : $(round(t_down*1e3, digits=2)) ms")
        println("matmul on GPU : $(round(t_mul*1e3, digits=2)) ms")
    end
end

#%% The matrix product, resident and complete
matmul_to_host(A, B) = CUDA.@sync Array(CuArray(A) * CuArray(B))

if gpu_available
    let
        N = 4096
        Acpu = rand(Float32, N, N); Bcpu = rand(Float32, N, N)
        Agpu = CuArray(Acpu); Bgpu = CuArray(Bcpu)
        reference = Acpu * Bcpu
        result = matmul_to_host(Acpu, Bcpu)
        rtol = 1e-4  # an example application requirement, not a universal error bound
        println("Float32 result meets rtol=$rtol: ", isapprox(result, reference; rtol=rtol))
        println("Relative error: ", norm(result-reference)/norm(reference))
        print("CPU, host result: "); @btime $Acpu * $Bcpu
        print("GPU, resident inputs/result: "); @btime CUDA.@sync $Agpu * $Bgpu
        print("GPU, complete host-to-host job: "); @btime matmul_to_host($Acpu, $Bcpu)
    end
end

#%% 5. GPU workload size — when does it start paying off?
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
# Compare resident timings with full host-to-host runtime. Test overhead rather than inferring a universal threshold.

#%% Batching on the GPU — time per batch and per example, resident inputs
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

#%% 6. Monte-Carlo π on the GPU — same count as before, 100 million points
# Everything stays on the card; only a scalar comes back.
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # drawn straight into VRAM
    y = CUDA.rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0    # 1f0 = 1 as a Float32
    return 4 * sum(inside) / n           # reduced on the GPU
end

function pi_cpu_vec(n)                   # mirrors pi_gpu, allocates
    x = rand(Float32, n)
    y = rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0
    return 4 * sum(inside) / n
end

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
# State the RNG/dtype, timed costs and statistical accuracy before interpreting speedup.

#%% 7. Optional — coalescing: locality again, at the scale of a warp (32 threads)
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
# The ratio may rise; index arithmetic and allocations also change.
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

#%% 8. Wrap-up
# Is there enough parallel work? Can intermediate data stay on the GPU?
# Is the complete job faster at the required accuracy?
