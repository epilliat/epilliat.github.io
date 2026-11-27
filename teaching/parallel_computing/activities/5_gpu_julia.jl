# π on the card (Julia)  30 min.
# Fill one blank, run each #%% cell.
# Needs an NVIDIA GPU + CUDA.jl:  using Pkg; Pkg.add("CUDA")  (the 1st `using CUDA` downloads the runtime, once)

#%% setup
using CUDA
gpu_available = CUDA.functional()
gpu_available || println("No GPU: complete gpu-fallback.html from the course page.")
gpu_available && println("GPU: ", CUDA.name(CUDA.device()))

# the dart count, on given points (1f0 = Float32 1.0)
count_in_disk(x, y) = sum(x.^2 .+ y.^2 .<= 1f0)
fixed_x = Float32[0, .5, 1, 1]
fixed_y = Float32[0, .5, 0, 1]
@assert count_in_disk(fixed_x, fixed_y) == 3
gpu_available && @assert count_in_disk(CuArray(fixed_x), CuArray(fixed_y)) == 3

# CPU version, Float32 like CUDA.rand
pi_cpu(n) = 4 * count_in_disk(rand(Float32, n), rand(Float32, n)) / n

#%% Your turn: the same π, but draw the points ON the GPU with CUDA.rand
function pi_gpu(n)
    #= SOLUTION: draw x and y using CUDA.rand(Float32, n), then apply count_in_disk and return 4*hits/n =#
    x = CUDA.rand(Float32, n)            # Float32, drawn straight into GPU memory
    y = CUDA.rand(Float32, n)
    return 4 * count_in_disk(x, y) / n
    #= END =#
end

#%% Try it
pi_gpu(1_000_000)

#%% One size, CPU vs GPU; then try 10_000 and 50_000_000
function fastest_time(f, args...)
    f(args...)                                  # warm up (the 1st GPU call compiles the kernel)
    times = [@elapsed(CUDA.@sync f(args...)) for _ in 1:5]   # @sync: wait for the GPU to finish
    return minimum(times)
end
fastest_time(pi_cpu, 1_000_000), fastest_time(pi_gpu, 1_000_000)

#%% The whole table
println("      points |  CPU (ms) |  GPU (ms) | speedup | approx. standard error")
for n in (10_000, 1_000_000, 50_000_000)
    tc = fastest_time(pi_cpu, n)
    tg = fastest_time(pi_gpu, n)
    println(lpad(n, 12), " | ", lpad(round(tc * 1e3, digits=2), 9), " | ",
            lpad(round(tg * 1e3, digits=2), 9), " | ", lpad(round(tc / tg, digits=1), 6), "× | ", round(1.64/sqrt(n), sigdigits=3))
end

# Report a standard-error target, not a guaranteed absolute-error bound.
# These times include RNG, allocation, counting/reduction, synchronization and scalar return.
# No point vectors are uploaded. State whether the tested n meets your accuracy requirement.

#%% Batching: n products A*x, or one product A*X (X holds n vectors, one per column)
using LinearAlgebra
n = 1024
A = CUDA.rand(Float32, n, n)
X = CUDA.rand(Float32, n, n)
Y = similar(X)                          # room for the n results
x = X[:, 1]

#%% One matrix-vector product
fastest_time(*, A, x)

#%% n matrix-vector products, one by one
function one_by_one!(Y, A, X)
    for j in axes(X, 2)
        mul!(view(Y, :, j), A, view(X, :, j))
    end
    return Y
end
fastest_time(one_by_one!, Y, A, X)

#%% Your turn: the same n products in one matrix-matrix product
function batched!(Y, A, X)
    #= SOLUTION: one call to mul! with the whole matrices =#
    mul!(Y, A, X)
    #= END =#
    return Y
end
@assert batched!(similar(X), A, X) ≈ one_by_one!(similar(X), A, X)
fastest_time(batched!, Y, A, X)

#%% The whole table
println("     n | one by one (ms) | batched (ms) | ratio")
for n in (256, 1024, 4096)
    An, Xn, Yn = CUDA.rand(Float32, n, n), CUDA.rand(Float32, n, n), CUDA.zeros(Float32, n, n)
    t1 = fastest_time(one_by_one!, Yn, An, Xn)
    tb = fastest_time(batched!, Yn, An, Xn)
    println(lpad(n, 6), " | ", lpad(round(t1 * 1e3, digits=2), 15), " | ", lpad(round(tb * 1e3, digits=2), 12), " | ", round(t1 / tb, digits=1), "×")
end
