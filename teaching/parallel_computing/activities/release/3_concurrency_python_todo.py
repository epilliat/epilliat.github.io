# Local version of 3_concurrency_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # A3a — overlap waiting — 20 minutes
# 20 archived daily closes from Binance, or labelled simulated waits if offline.

# %%
import urllib.request, zipfile, io, time, os
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

# Set COURSE_LIVE_IO=0 before starting the kernel for the offline route.
LIVE_IO = os.environ.get("COURSE_LIVE_IO", "1") == "1"
days = [date(2024, 6, 1) + timedelta(d) for d in range(20)]

def close_of(day):
    if not LIVE_IO:
        time.sleep(0.05)  # controlled waiting, NOT a network benchmark
        return 100.0 + (day - days[0]).days  # synthetic values, NOT Bitcoin prices
    url = f'https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-{day}.zip'
    with urllib.request.urlopen(url, timeout=3) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        row = archive.read(archive.namelist()[0]).decode().splitlines()[0]
    return float(row.split(',')[4])

def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start

# One probe: on failure use the controlled-wait exercise for the whole comparison.
if LIVE_IO:
    try:
        close_of(days[0])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print("Archive unavailable:", type(error).__name__)
        LIVE_IO = False
print("REAL archived prices" if LIVE_IO else "SIMULATED waits + synthetic values")

# %%
# One day
timed(lambda: close_of(days[0]))[1]

# %%
# Two days, one after the other
timed(lambda: [close_of(days[0]), close_of(days[1])])[1]

# %%
# Two days at once
def two_at_once():
    with ThreadPoolExecutor() as pool:
        a = pool.submit(close_of, days[0])
        b = pool.submit(close_of, days[1])
        return a.result(), b.result()

timed(two_at_once)[1]

# %% [markdown]
# Your turn: the pool map, results in input order.

# %%
def download_all(days):
    # ✏️ your turn: map close_of over days using a ThreadPoolExecutor; wait for the results
    raise NotImplementedError("remove this line and write your code")

# %%
# If a later download fails, restart BOTH measurements in simulation mode.
try:
    closes_seq, t_seq = timed(lambda: [close_of(d) for d in days])
    closes, t_conc = timed(lambda: download_all(days))
except (OSError, ValueError, zipfile.BadZipFile) as error:
    LIVE_IO = False
    print("Restarting comparison: SIMULATED waits + synthetic values", type(error).__name__)
    closes_seq, t_seq = timed(lambda: [close_of(d) for d in days])
    closes, t_conc = timed(lambda: download_all(days))
assert closes == closes_seq
print("real archive" if LIVE_IO else "simulation", f"seq={t_seq:.3f}s concurrent={t_conc:.3f}s ratio={t_seq/t_conc:.2f}")
