# π on the GPU — but only if you feed it.  ~5 min.
# The running example ends here: the same Monte-Carlo count of darts in the disk, now on
# the GPU. Fill one blank, run, read the table.
#
# Needs an NVIDIA GPU + CUDA.jl:  using Pkg; Pkg.add("CUDA")
# (The first `using CUDA` downloads the CUDA runtime — a few minutes, once.)
# This activity writes no kernel: the same array code runs on the card by multiple dispatch.

#%% setup
using CUDA
@assert CUDA.functional() "No usable GPU — this activity needs an NVIDIA card with CUDA."
println("GPU: ", CUDA.name(CUDA.device()))

# the dart count again, on points we are GIVEN (1f0 = the Float32 literal 1.0)
count_in_disk(x, y) = sum(x.^2 .+ y.^2 .<= 1f0)

# CPU version: Float32 points, like the GPU (CUDA.rand draws Float32) — a fair comparison
pi_cpu(n) = 4 * count_in_disk(rand(Float32, n), rand(Float32, n)) / n

#%% Your turn: the same π, but draw the points ON the GPU with CUDA.rand
function pi_gpu(n)
    # ✏️ your turn: x = CUDA.rand(n); y = CUDA.rand(n); then the SAME count_in_disk as pi_cpu
    error("remove this line and write your code")
end

#%% Run me — the same π, three problem sizes, CPU vs GPU
# warm up + take the fastest of a few runs (the 1st GPU call compiles the kernel and
# spins the card up from idle — otherwise the smallest size looks absurdly slow)
timecpu(n) = (pi_cpu(n); minimum(@elapsed(pi_cpu(n)) for _ in 1:5))
timegpu(n) = (pi_gpu(n); minimum(CUDA.@elapsed(pi_gpu(n)) for _ in 1:5))
timegpu(50_000_000)                        # thorough warm-up before we measure anything

println("      points |  CPU (ms) |  GPU (ms) | speedup")
for n in (10_000, 1_000_000, 50_000_000)
    tc = timecpu(n)
    tg = timegpu(n)
    println(lpad(n, 12), " | ", lpad(round(tc * 1e3, digits=2), 9), " | ",
            lpad(round(tg * 1e3, digits=2), 9), " | ", lpad(round(tc / tg, digits=1), 6), "×")
end

# What you just saw:
# At 10_000 points the GPU is ~1× or slower — you paid to launch work on the card, for
# almost no work. At millions of points it's tens of ×. A GPU has thousands of tiny cores;
# they only pay off when there's enough work to keep them busy. FEED THE GPU.
#
# The SAME count_in_disk ran on both: x and y are CuArrays on the GPU, so MULTIPLE DISPATCH
# (the dispatch module) picked the GPU methods. Same source, different hardware.
