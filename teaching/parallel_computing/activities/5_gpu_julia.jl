# A6 (Julia) — π on the GPU.  30 min.
# Fill one blank, run each #%% cell.
# Needs an NVIDIA GPU + CUDA.jl:  using Pkg; Pkg.add("CUDA")  (the 1st `using CUDA` downloads the runtime, once)

#%% setup
using CUDA
gpu_available = CUDA.functional()
gpu_available || println("No GPU: complete gpu-fallback.html from the course page.")
gpu_available && println("GPU: ", CUDA.name(CUDA.device()))

# the dart count, on given points (1f0 = Float32 1.0)
count_in_disk(x, y) = sum(x.^2 .+ y.^2 .<= 1f0)
@assert count_in_disk(Float32[0, .5, 1, 1], Float32[0, .5, 0, 1]) == 3

# CPU version, Float32 like CUDA.rand
pi_cpu(n) = 4 * count_in_disk(rand(Float32, n), rand(Float32, n)) / n

#%% Your turn: the same π, but draw the points ON the GPU with CUDA.rand
function pi_gpu(n)
    #= SOLUTION: x = CUDA.rand(n); y = CUDA.rand(n); then the SAME count_in_disk as pi_cpu =#
    x = CUDA.rand(n)            # Float32, drawn straight into GPU memory
    y = CUDA.rand(n)
    return 4 * count_in_disk(x, y) / n
    #= END =#
end

#%% Try it
pi_gpu(1_000_000)

#%% One size, CPU vs GPU; then try 10_000 and 50_000_000
# warm up + fastest of 5 runs (the 1st GPU call compiles the kernel and wakes the card up)
timecpu(n) = (pi_cpu(n); minimum(@elapsed(pi_cpu(n)) for _ in 1:5))
timegpu(n) = (CUDA.@sync pi_gpu(n); minimum(@elapsed(CUDA.@sync pi_gpu(n)) for _ in 1:5))
timegpu(50_000_000)                        # thorough warm-up before we measure anything
(cpu = timecpu(1_000_000), gpu = timegpu(1_000_000))

#%% The whole table
if gpu_available

println("      points |  CPU (ms) |  GPU (ms) | speedup")
for n in (10_000, 1_000_000, 50_000_000)
    tc = timecpu(n)
    tg = timegpu(n)
    println(lpad(n, 12), " | ", lpad(round(tc * 1e3, digits=2), 9), " | ",
            lpad(round(tg * 1e3, digits=2), 9), " | ", lpad(round(tc / tg, digits=1), 6), "×")
end

end
