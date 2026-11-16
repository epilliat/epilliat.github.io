# Local version of 5_gpu_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # A6 — π on a GPU — 30 minutes
# No GPU? Use the GPU fallback page of the course.

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
if gpu:
    assert count_in_disk(x.cuda(), y.cuda()).item() == 3

# %% [markdown]
# Your turn: `pi_gpu`, the same computation with the draws on the GPU (`torch.rand(n, device='cuda')`).

# %%
def pi_gpu(n):
    # ✏️ your turn: draw x and y with device='cuda', same count_in_disk, then torch.cuda.synchronize()
    raise NotImplementedError("remove this line and write your code")

# %%
# Try it
pi_gpu(1_000_000) if gpu else "no GPU"

# %%
# One size, CPU vs GPU; then try 10_000 and 50_000_000
def best(f, n, iters=10):
    f(n)                              # warm up (1st GPU call compiles / allocates)
    t0 = time.perf_counter()
    for _ in range(iters): p = f(n)
    return (time.perf_counter() - t0) / iters, p

(best(pi_cpu, 1_000_000)[0], best(pi_gpu, 1_000_000)[0]) if gpu else "no GPU"

# %%
# The whole table
if gpu:
    print(f"{'points':>12} | {'CPU':>9} | {'GPU':>9} | speedup")
    for n in (10_000, 1_000_000, 50_000_000):
        tc, _ = best(pi_cpu, n)
        tg, p = best(pi_gpu, n)
        print(f'{n:>12,} | {tc*1e3:7.2f}ms | {tg*1e3:7.2f}ms | {tc/tg:5.1f}×   (π≈{p:.4f})')
else:
    print("No GPU measurements. Complete gpu-fallback.html from the course page.")
