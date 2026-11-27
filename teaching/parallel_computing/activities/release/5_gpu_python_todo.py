# Local version of 5_gpu_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # π on the card — 30 minutes
# No GPU? Use the GPU fallback page of the course.
#
# Compare the complete estimator at a stated statistical-accuracy target. Timings are the mean per completed call after warmup; fixed-point checks test the counting logic separately.

# %%
import torch, time
gpu = torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0) if gpu else 'unavailable — use the fallback page')

def count_in_disk(x, y):                          # how many points land in the quarter disk
    return ((x*x + y*y) <= 1.0).sum()             # runs wherever x and y live (CPU or GPU)

def pi_cpu(n):
    x = torch.rand(n); y = torch.rand(n)          # n random points in the unit square (float32)
    return 4 * count_in_disk(x, y).item() / n
# Same fixed points, known count: a correctness check, not a π accuracy test.
x = torch.tensor([0., 0.5, 1., 1.], dtype=torch.float32)
y = torch.tensor([0., 0.5, 0., 1.], dtype=torch.float32)
assert count_in_disk(x, y).item() == 3
assert count_in_disk(x.cuda(), y.cuda()).item() == 3

# %% [markdown]
# Your turn: `pi_gpu`, the same computation with the draws on the GPU (`torch.rand(n, device='cuda')`).

# %%
def pi_gpu(n):
    # ✏️ your turn: draw x and y with device='cuda', same count_in_disk, then torch.cuda.synchronize()
    raise NotImplementedError("remove this line and write your code")

# %%
# Try it
pi_gpu(1_000_000)

# %%
# One size, CPU vs GPU; then try 10_000 and 50_000_000
def mean_time(f, n, iters=10):
    f(n)                              # warm up (1st GPU call compiles / allocates)
    t0 = time.perf_counter()
    for _ in range(iters): p = f(n)
    return (time.perf_counter() - t0) / iters, p

(mean_time(pi_cpu, 1_000_000)[0], mean_time(pi_gpu, 1_000_000)[0])

# %%
# The whole table
print(f"{'points':>12} | {'CPU':>9} | {'GPU':>9} | speedup")
for n in (10_000, 1_000_000, 50_000_000):
    tc, _ = mean_time(pi_cpu, n)
    tg, p = mean_time(pi_gpu, n)
    print(f'{n:>12,} | {tc*1e3:7.2f}ms | {tg*1e3:7.2f}ms | {tc/tg:5.1f}×   (π≈{p:.4f}, SE≈{1.64/n**0.5:.2g})')

# %% [markdown]
# **Decision:** is there enough work, do intermediate arrays stay on the GPU, and is the complete estimator faster at your accuracy target? SE ≈ 1.64/√n is not a guaranteed absolute-error bound. No point vectors are uploaded here. Record allocation, RNG, reduction, synchronization and scalar-return costs; do not compare random CPU/GPU estimates by exact equality.

# %% [markdown]
# ### Batching: n products `A @ x`, or one product (X holds n vectors, one per row)

# %%
n = 1024
A = torch.rand(n, n, device='cuda')
X = torch.rand(n, n, device='cuda')

def best(f):
    f(); torch.cuda.synchronize()                  # warm up
    times = []
    for _ in range(5):
        t0 = time.perf_counter(); f(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)

# %%
# One matrix-vector product
best(lambda: A @ X[0])

# %%
# n matrix-vector products, one by one
def one_by_one(A, X):
    return torch.stack([A @ X[j] for j in range(X.shape[0])])

best(lambda: one_by_one(A, X))

# %% [markdown]
# Your turn: the same n products in one matrix-matrix product.

# %%
def batched(A, X):
    # ✏️ your turn: one product of X with the transpose of A (row j of the result = A @ X[j])
    raise NotImplementedError("remove this line and write your code")

assert torch.allclose(batched(A, X), one_by_one(A, X), rtol=1e-4)
best(lambda: batched(A, X))

# %%
# The whole table
print(f"{'n':>6} | {'one by one':>12} | {'batched':>9} | ratio")
for n in (256, 1024, 4096):
    A = torch.rand(n, n, device='cuda'); X = torch.rand(n, n, device='cuda')
    t1, tb = best(lambda: one_by_one(A, X)), best(lambda: batched(A, X))
    print(f"{n:>6} | {t1*1e3:9.2f} ms | {tb*1e3:6.2f} ms | {t1/tb:5.1f}×")
