# Local version of 3_server_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # A3d — a slow server — 20 minutes
# No Internet needed: this notebook starts its own web servers (0.2 s per answer).

# %%
# 0. Setup (given): a slow server, and a function that fetches one URL
import http.server, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

class Slow(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(0.2)                    # the server takes its time
        body = (str(self.server.number) if self.path == "/number" else self.path).encode()  # echoes the path
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):          # quiet
        pass

class OneAtATime(http.server.HTTPServer):
    request_queue_size = 64
class OnePerRequest(http.server.ThreadingHTTPServer):  # one thread per request
    request_queue_size = 64

def start(server_class, host="127.0.0.1", number=0):
    server = server_class((host, 0), Slow)   # port 0 = any free port
    server.number = number
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"

def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()

def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0

concurrent = start(OnePerRequest)
serial = start(OneAtATime)

# %%
# 1. One request
timed(lambda: fetch(concurrent + "/1"))

# %%
# 2. Two requests, one after the other
timed(lambda: [fetch(concurrent + "/1"), fetch(concurrent + "/2")])

# %%
# 3. Two requests at once
def two_at_once(server):
    with ThreadPoolExecutor() as pool:
        a = pool.submit(fetch, server + "/1")
        b = pool.submit(fetch, server + "/2")
        return a.result(), b.result()

timed(lambda: two_at_once(concurrent))

# %%
# 4. Same client code, the other server
timed(lambda: two_at_once(serial))

# %% [markdown]
# 5. Now 20 requests with `workers` threads: complete `fetch_all`.

# %%
def fetch_all(urls, workers):
    # ✏️ your turn: map fetch over urls with `workers` threads; return a list
    raise NotImplementedError("remove this line and write your code")

urls = [concurrent + f"/{i}" for i in range(1, 21)]
fetch_all(urls, 2) == [f"/{i}" for i in range(1, 21)]   # right answers, in order

# %%
# 6. Play: change 5 to 1, 20…, and `concurrent` to `serial` in `urls`
timed(lambda: fetch_all(urls, 5))

# %%
# 7. The whole table
for name, server in [("serial", serial), ("concurrent", concurrent)]:
    for workers in (1, 5, 20):
        t = timed(lambda: fetch_all([server + f"/{i}" for i in range(1, 21)], workers))
        print(f"{name}  workers={workers:2}  {t:.2f} s")

# %% [markdown]
# ### Optional — the class sum
# On your laptop, not Colab. Your server answers `my_number` at `/number`; register it, get everyone's address, return the sum. Nobody reaches the registry? School Wi-Fi may block it: try a phone hotspot.

# %%
my_number = 7                                   # your choice
shared = start(OnePerRequest, host="0.0.0.0", number=my_number)
registry = "http://127.0.0.1:9000"              # replace by the address on the board
fetch(registry + "/register?port=" + shared.rsplit(":", 1)[1])   # your address, as the others see it

# %%
addresses = fetch(registry + "/list").split()   # everyone's addresses (Python or Julia servers)
# ✏️ your turn: fetch "/number" from every address at once with fetch_all, convert to int, sum
raise NotImplementedError("remove this line and write your code")
