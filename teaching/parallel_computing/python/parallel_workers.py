"""Importable CPU workers: no pool or work is started during import."""
import random


def cpu_load(n):
    return sum((i % 7) * 1.0000001 for i in range(n))


def count_in_disk(job):
    """One fixed (number of points, seed) lot, using pure Python arithmetic."""
    n, seed = job
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        x, y = rng.random(), rng.random()
        hits += x*x + y*y <= 1.0
    return hits


def make_jobs(n, lots=8, seed=2026):
    """Keep lots fixed when changing workers; include the remainder."""
    if n < 1 or lots < 1:
        raise ValueError("n and lots must be positive")
    q, r = divmod(n, lots)
    seeds = random.Random(seed)
    return [(q + (i < r), seeds.getrandbits(128)) for i in range(lots)]
