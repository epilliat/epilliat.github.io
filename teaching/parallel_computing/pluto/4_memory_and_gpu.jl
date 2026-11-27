### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ aa07d25d-4f02-4013-95a0-49f72ddf7323
md"""
# Memory and GPU

Loop order and batch size affect access patterns and data reuse. The memory chapter covers
1. repeated sums and processing rate, 2. loop order, and 3. batching. Layout adaptation,
output-buffer reuse and arithmetic intensity are optional references.
The GPU chapter follows in sections 4–7. Use selected demonstrations from the session plan.
"""

# ╔═╡ ee0a05ee-935e-42c3-b567-5624fb902747
md"""
## 0. Getting started

Needs BenchmarkTools and CUDA.jl, even without a card: the GPU cells then skip.
"""

# ╔═╡ 3b62bc4c-9765-4424-b7ee-2800c085103a
begin
    using BenchmarkTools
    using LinearAlgebra
    using CUDA                 # needs the package; a card is optional
    md"Packages loaded ✓"
end

# ╔═╡ c3790242-0bd4-4168-858a-3e58d5bb3134
gpu_available = CUDA.functional()

# ╔═╡ e60ad78b-4669-47b1-b819-443e4454f585
md"""
## 1. Repeated sums as the array grows

Illustrative access latencies: L1 ~1 ns, L2 ~4 ns, L3 ~15 ns, RAM ~100 ns. Capacities and
sharing vary by hardware. Nearby values support spatial locality; reusing recently accessed
values supports temporal locality.

Time repeated sums over increasing footprints. Report elapsed time divided by the number of
elements processed, not individual-load latency. Sequential reads can overlap memory transfers.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c002
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

# ╔═╡ 58cf2a8a-433f-4737-8116-044f7f8eada1
md"""
## 2. Locality — the same sum, walked two ways

Same additions, two loop orders.
"""

# ╔═╡ 3218651a-5b26-42d1-b330-917d0801ed44
A = rand(8_000, 8_000)   # 512 MB: larger than the reference laptop’s caches

# ╔═╡ 2126f361-6be9-4f14-a358-6a034213fa4f
function sum_by_columns(M)
    s = zero(eltype(M))
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)             # 1st index innermost
            s += M[i, j]
        end
    end
    return s
end

# ╔═╡ 6908e575-a27a-43b8-b7ee-2cf0bce7f8fc
function sum_by_rows(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)
        for j in axes(M, 2)             # 2nd index innermost
            s += M[i, j]
        end
    end
    return s
end

# ╔═╡ 43f403f1-a036-4454-8670-327d079dd5cc
@assert sum_by_columns(A) ≈ sum_by_rows(A)

# ╔═╡ cdad600b-7259-4388-a83b-0a2f42f83aa1
let
    print("columns : "); @btime sum_by_columns($A)
    print("rows    : "); @btime sum_by_rows($A)
end

# ╔═╡ 09844b2e-6544-494d-a7c1-50c3d5b9c5d6
md"""
**Test the effect of matrix size:** predict how the ratio changes as matrices get smaller.
A smaller difference when the data fits in cache supports a locality explanation, without
isolating cache, prefetching and translation effects or implying that order can never matter.
"""

# ╔═╡ 01d52dcd-e327-418c-9906-dddd778b01f0
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

# ╔═╡ 8e4c45e6-5694-4af9-804c-4de21952feeb
md"""
**Optional — reuse output buffers.** Compare time and allocated bytes.
Reusing z avoids allocating an output, but the calculation still reads x/y and writes z.
`mul!(C,A,B)` similarly reuses C. Whether reuse improves runtime depends on the operation,
output size and allocation strategy; measure rather than assume.
"""

# ╔═╡ 9a99b467-6b9a-4353-9411-01f0e610b3e5
let
    n = 10_000_000
    x = rand(n); y = rand(n); z = zeros(n)
    print("allocates : "); @btime 2 .* $x .+ $y
    print("in place  : "); @btime $z .= 2 .* $x .+ $y
end

# ╔═╡ 6d24a0f2-932f-47e7-a262-68c623e9ac00
let
    n = 2000
    Ai = rand(n, n); Bi = rand(n, n); Ci = zeros(n, n)
    print("C = A * B     : "); @btime $Ai * $Bi
    print("mul!(C, A, B) : "); @btime mul!($Ci, $Ai, $Bi)
end

# ╔═╡ ad57f7bf-d863-4be8-b5d4-d0dfd0603b69
md"""
### Optional — arithmetic intensity: FLOP per byte

Use GFLOP/s to compare computational throughput, and elapsed time for application latency.
Approximate Float64 intensity is 0.25 FLOP/byte for matrix-vector multiplication and n/12
for square matrix multiplication under ideal three-matrix traffic. Actual traffic depends
on implementation and cache reuse; the model is not a bandwidth measurement.
"""

# ╔═╡ 05884f27-4254-4a8e-8177-e38454dc8aac
begin
    n = 2000
    Mat = rand(n, n)
    xvec = rand(n)
    Mat2 = rand(n, n)
    md"Data ready."
end

# ╔═╡ dc815f82-3a2e-40a7-b9a1-67daf6c38766
let
    print("matrix × vector : "); @btime $Mat * $xvec    # 2n² FLOP
    print("matrix × matrix : "); @btime $Mat * $Mat2    # 2n³ FLOP
end

# ╔═╡ ba7c1100-0001-4a01-8b01-aaaaaaaa0001
md"""
## 3. Batching: one layer `W`, `B` examples stacked as columns of `X`

Compare `W*x` with `W*X`. Matrix multiplication can reuse weights across examples; measure time per example. (Run before the GPU cells: CPU/GPU power and thermal state can affect laptop timings.)
"""

# ╔═╡ ba7c1100-0002-4a01-8b01-aaaaaaaa0002
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

# ╔═╡ 92cd196c-2cab-4aee-aecb-89f06133486c
md"""
## 4. Device and precision

CuArray preserves the input element type while copying to device memory; cu can narrow
Float64 to Float32. Keep the inputs and accuracy requirement explicit. A GPU thread is not
a CPU core. The following cells run only if a CUDA device is available.
"""

# ╔═╡ 7f4abed0-2b61-454d-922d-2d8eccad9c46
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

# ╔═╡ 71089296-a3b6-4f51-8ecc-a4909bc66388
md"""
**What the copy costs:** time uploads, resident computation and downloads separately.
Transfer calls also allocate. Their sum is only a model of complete runtime; measure the
complete path directly. An application requiring a full matrix must include its download.
"""

# ╔═╡ c6aa25cd-9a29-4cf6-afab-ea848a9be746
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

# ╔═╡ bf891e84-9d69-454a-a77b-724596266414
md"""
**The matrix product, resident and complete.** Use identical Float32 inputs and check the
returned host matrix against the CPU result. The example tolerance is a stated requirement,
not a guarantee. Compare CPU time, resident GPU time and the complete host-to-host job.
"""

# ╔═╡ b5d50000-0001-4000-8000-000000000001
matmul_to_host(A, B) = CUDA.@sync Array(CuArray(A) * CuArray(B))

# ╔═╡ 78901ba1-a4f3-4808-99ed-b151bab95b78
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

# ╔═╡ 1ed9c688-150c-423c-a086-181603547e2c
md"""
## 5. GPU workload size — when does it start paying off?


"""

# ╔═╡ 36c37805-567f-4baf-b8d3-5d42d5027b88
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

# ╔═╡ 7663fb47-1f0c-4383-a5dd-e55f9a1b5abc
md"""
Predict small versus large resident products. Launch overhead can matter for small work;
transfers are excluded here. The table alone does not identify every large-size limitation.
Record dimensions, dtype, resource counts and the runtime's numerical configuration.
"""

# ╔═╡ ba7c1100-0004-4a01-8b01-aaaaaaaa0004
md"""
### Batching on the GPU

Read total time and time per example. Resident inputs exclude transfer costs. Low throughput
at B=1 is not a measured utilization percentage; choose B subject to latency and memory limits.
"""

# ╔═╡ ba7c1100-0005-4a01-8b01-aaaaaaaa0005
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

# ╔═╡ cb000000-0000-4c00-9a00-00000000c005
md"""
## 7. Optional — coalescing: locality again, at the scale of a warp (32 threads)

Same reads, different order. **Why must the index be a permutation** (`gcd(s, N) = 1`)?
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c006
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

# ╔═╡ cb000000-0000-4c00-9a00-00000000c007
md"""
Smaller ratio than you expected? **Which other bytes does `a[idx]` read and write?**
"""

# ╔═╡ f526773a-c8d8-46b7-b0b7-165a48f6417f
md"""
**Test that explanation:** compute the index in the kernel (no `idx` array to read). The ratio may **rise**; index arithmetic and allocations also change.
"""

# ╔═╡ 18ecadcb-0926-4ef7-9b4a-547d57203342
function gather_kernel!(out, a, s, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        j = ((i - 1) * s) % n + 1        # index COMPUTED, never loaded
        @inbounds out[i] = a[j]
    end
    return nothing
end

# ╔═╡ 5aa43061-0c7e-4193-a1de-a4ff25d5d0ee
gather_run!(out, a, s, n) =
    CUDA.@sync @cuda threads=256 blocks=cld(n, 256) gather_kernel!(out, a, s, n)

# ╔═╡ 1bb4c950-c705-4c0e-8833-6636b3892e26
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

# ╔═╡ e85db6c9-b322-4a2f-aed6-295f21cb362d
md"""
## 6. Monte-Carlo π on the GPU — same count as before, 100 million points

Everything stays on the card; only a scalar comes back.
"""

# ╔═╡ 7a6d400a-f826-4b67-9a7d-717351daa9bf
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # drawn straight into VRAM
    y = CUDA.rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0    # 1f0 = 1 as a Float32
    return 4 * sum(inside) / n           # reduced on the GPU
end

# ╔═╡ aa59c041-e5fa-4cb0-a8f0-43c353255569
function pi_cpu_vec(n)                   # mirrors pi_gpu, allocates
    x = rand(Float32, n)
    y = rand(Float32, n)
    inside = (x.^2 .+ y.^2) .<= 1f0
    return 4 * sum(inside) / n
end

# ╔═╡ 324e4aa4-4012-4326-85bd-f41babd9ef0d
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

# ╔═╡ 798cfc8b-51b7-4086-8b30-faa6be29c299
let
    n = 100_000_000
    print("CPU scalar loop : "); @btime 4 * count_in_disk($n) / $n
    print("CPU vectorized  : "); @btime pi_cpu_vec($n)
    if gpu_available
        println("pi_gpu = ", pi_gpu(n), "   (π = ", Float64(π), ")")
        print("GPU             : "); @btime CUDA.@sync pi_gpu($n)
    end
end

# ╔═╡ 0d3be671-416b-4c2f-a605-1995a602e132
md"""
State RNGs, precision, allocations and timed scope before comparing π implementations.
Check fixed points for the count logic and a statistical error target for the estimator.
Different RNGs mean CPU/GPU estimates need not be bitwise equal. A parallel reduction has
less parallelism in its final stages; it is not entirely sequential.
"""

# ╔═╡ 130f0c20-5ca5-4045-9e70-e3898789bf01
md"""
## 8. GPU decision

- Is there enough parallel work?
- Can intermediate data stay on the GPU?
- Is the complete job faster at the required accuracy?

Return to the course synthesis: diagnose, choose a change, validate and measure the whole program.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─aa07d25d-4f02-4013-95a0-49f72ddf7323
# ╟─ee0a05ee-935e-42c3-b567-5624fb902747
# ╠═3b62bc4c-9765-4424-b7ee-2800c085103a
# ╠═c3790242-0bd4-4168-858a-3e58d5bb3134
# ╟─e60ad78b-4669-47b1-b819-443e4454f585
# ╠═cb000000-0000-4c00-9a00-00000000c002
# ╠═cb000000-0000-4c00-9a00-00000000c003
# ╟─58cf2a8a-433f-4737-8116-044f7f8eada1
# ╠═3218651a-5b26-42d1-b330-917d0801ed44
# ╠═2126f361-6be9-4f14-a358-6a034213fa4f
# ╠═6908e575-a27a-43b8-b7ee-2cf0bce7f8fc
# ╠═43f403f1-a036-4454-8670-327d079dd5cc
# ╠═cdad600b-7259-4388-a83b-0a2f42f83aa1
# ╟─09844b2e-6544-494d-a7c1-50c3d5b9c5d6
# ╠═01d52dcd-e327-418c-9906-dddd778b01f0
# ╟─ba7c1100-0001-4a01-8b01-aaaaaaaa0001
# ╠═ba7c1100-0002-4a01-8b01-aaaaaaaa0002
# ╟─8e4c45e6-5694-4af9-804c-4de21952feeb
# ╠═9a99b467-6b9a-4353-9411-01f0e610b3e5
# ╠═6d24a0f2-932f-47e7-a262-68c623e9ac00
# ╟─ad57f7bf-d863-4be8-b5d4-d0dfd0603b69
# ╠═05884f27-4254-4a8e-8177-e38454dc8aac
# ╠═dc815f82-3a2e-40a7-b9a1-67daf6c38766
# ╟─92cd196c-2cab-4aee-aecb-89f06133486c
# ╠═7f4abed0-2b61-454d-922d-2d8eccad9c46
# ╟─71089296-a3b6-4f51-8ecc-a4909bc66388
# ╠═c6aa25cd-9a29-4cf6-afab-ea848a9be746
# ╟─bf891e84-9d69-454a-a77b-724596266414
# ╠═b5d50000-0001-4000-8000-000000000001
# ╠═78901ba1-a4f3-4808-99ed-b151bab95b78
# ╟─1ed9c688-150c-423c-a086-181603547e2c
# ╠═36c37805-567f-4baf-b8d3-5d42d5027b88
# ╟─7663fb47-1f0c-4383-a5dd-e55f9a1b5abc
# ╟─ba7c1100-0004-4a01-8b01-aaaaaaaa0004
# ╠═ba7c1100-0005-4a01-8b01-aaaaaaaa0005
# ╟─e85db6c9-b322-4a2f-aed6-295f21cb362d
# ╠═7a6d400a-f826-4b67-9a7d-717351daa9bf
# ╠═aa59c041-e5fa-4cb0-a8f0-43c353255569
# ╠═324e4aa4-4012-4326-85bd-f41babd9ef0d
# ╠═798cfc8b-51b7-4086-8b30-faa6be29c299
# ╟─0d3be671-416b-4c2f-a605-1995a602e132
# ╟─cb000000-0000-4c00-9a00-00000000c005
# ╠═cb000000-0000-4c00-9a00-00000000c006
# ╟─cb000000-0000-4c00-9a00-00000000c007
# ╟─f526773a-c8d8-46b7-b0b7-165a48f6417f
# ╠═18ecadcb-0926-4ef7-9b4a-547d57203342
# ╠═5aa43061-0c7e-4193-a1de-a4ff25d5d0ee
# ╠═1bb4c950-c705-4c0e-8833-6636b3892e26
# ╟─130f0c20-5ca5-4045-9e70-e3898789bf01
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
