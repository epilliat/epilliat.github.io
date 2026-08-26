# π on the GPU — but only if you feed it.  ~5 min.
# The course's red thread ends here: π by Monte-Carlo went sequential (module 1) → threads
# (module 3) → GPU (now). Fill one blank, run, read the table.
#
# Needs an NVIDIA GPU + CUDA.jl:  using Pkg; Pkg.add("CUDA")
# (The first `using CUDA` downloads the CUDA runtime — a few minutes, once.)
# We write NO kernel — the same array code runs on the card by multiple dispatch.

#%% setup
using CUDA, Printf
@assert CUDA.functional() "No usable GPU — this activity needs an NVIDIA card with CUDA."
println("GPU: ", CUDA.name(CUDA.device()))

pi_cpu(n) = (x = rand(n); y = rand(n); 4 * count(x.^2 .+ y.^2 .<= 1) / n)

#%% Your turn: the same π, but draw the points ON the GPU with CUDA.rand
function pi_gpu(n)
    #= SOLUTION: x = CUDA.rand(n); y = CUDA.rand(n); then the SAME count as pi_cpu =#
    x = CUDA.rand(n)
    y = CUDA.rand(n)
    return 4 * count(x.^2 .+ y.^2 .<= 1) / n
    #= END =#
end

#%% Run me — the same π, three problem sizes, CPU vs GPU
# warm up + take the fastest of a few runs (the 1st GPU call compiles the kernel and
# spins the card up from idle — otherwise the smallest size looks absurdly slow)
timecpu(n) = (pi_cpu(n); minimum(@elapsed(pi_cpu(n)) for _ in 1:5))
timegpu(n) = (pi_gpu(n); minimum(CUDA.@elapsed(pi_gpu(n)) for _ in 1:5))
timegpu(50_000_000)                        # thorough warm-up before we measure anything

@printf("%12s | %9s | %9s | speedup\n", "points", "CPU", "GPU")
for n in (10_000, 1_000_000, 50_000_000)
    tc = timecpu(n)
    tg = timegpu(n)
    @printf("%12d | %7.2fms | %7.2fms | %5.1f×\n", n, tc*1e3, tg*1e3, tc/tg)
end

# What you just saw:
# At 10_000 points the GPU is ~1× or slower — you paid to ship data to the card and launch
# it, for almost no work. At millions of points it's tens of ×. A GPU has thousands of tiny
# cores; they only pay off when there's enough work to keep them busy. FEED THE GPU.
#
# No kernel here: the SAME `count(x.^2 .+ y.^2 .<= 1)` as the CPU version. It ran on the GPU
# because x and y are CuArrays — MULTIPLE DISPATCH (module 2) picked the GPU methods. Same
# source, different hardware. The red thread is complete: π — sequential → threads → GPU.
