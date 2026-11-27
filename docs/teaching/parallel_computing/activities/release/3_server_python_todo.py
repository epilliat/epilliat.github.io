# Local version of 3_server_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # A slow server — 20 minutes
# No Internet needed: this notebook starts its own web servers (0.2 s per answer).

# %%
# 0. Setup (given): a slow server, and a function that fetches one URL
import http.server, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

class Slow(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(0.2)                    # the server takes its time
        answers = {"/inside": str(int(self.server.inside)), "/name": self.server.name}
        body = answers.get(self.path, self.path).encode()   # otherwise: echoes the path
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

def start(server_class, host="127.0.0.1", inside=False, name=""):
    server = server_class((host, 0), Slow)   # port 0 = any free port
    server.inside, server.name = inside, name
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
# ### Optional — π by the whole class
# On your laptop, not Colab. One random point each: your server answers at `/inside` (1 in the quarter disk, 0 outside) and `/name`; register it, get everyone's address, estimate π. Nobody reaches the registry? School Wi-Fi may block it: try a phone hotspot.

# %%
import random
x, y = random.random(), random.random()
inside = x*x + y*y <= 1                         # in the quarter disk?
my_name = ""                                    # optional: a name or a message
shared = start(OnePerRequest, host="0.0.0.0", inside=inside, name=my_name)
registry = "http://127.0.0.1:9000"              # replace by the address on the board
fetch(registry + "/register?port=" + shared.rsplit(":", 1)[1])   # your address, as the others see it

# %%
addresses = fetch(registry + "/list").split()   # everyone's addresses (Python or Julia servers)
# ✏️ your turn: fetch "/inside" from every address at once with fetch_all, convert to int, sum
raise NotImplementedError("remove this line and write your code")
4 * hits / len(addresses)

# %%
# Everyone's names or messages
list(zip(addresses, fetch_all([a + "/name" for a in addresses], len(addresses))))

# %%
# Instructor: the class board. Run it, then open http://localhost:9000/ (run by a student: a board of their own)
import html, socket, urllib.parse

def start_registry(port=9000):
    servers = {}                                   # IP -> address of its server
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))           # sends nothing: finds this laptop's IP
        me = s.getsockname()[0]

    def ask(url):
        try:
            return fetch(url)
        except Exception:                          # gone, or not answering
            return None

    def ask_all(addresses):                        # (name, point) of every server, all at once
        with ThreadPoolExecutor(max_workers=max(1, len(addresses))) as pool:
            return list(pool.map(lambda a: (ask(a + "/name") or "", ask(a + "/inside")), addresses))

    def board():
        rows = ask_all(list(servers.values()))
        points = [p for _, p in rows if p in ("0", "1")]
        hits = points.count("1")
        pi_hat = f"{4 * hits / len(points):.3f}" if points else "?"
        lines = "".join(f"<tr><td>{ {'1': '&#9679; inside', '0': '&#9675; outside'}.get(p, 'no answer') }</td>"
                        f"<td>{html.escape(n[:40]) + ('&hellip;' if len(n) > 40 else '')}</td></tr>" for n, p in rows)
        return f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<body style="font: 26px sans-serif; margin: 40px"><p>registry = <b>"http://{me}:{port}"</b></p>
<h1>&pi; &asymp; {pi_hat} <small>({len(points)} points, {hits} inside)</small></h1>
<table cellpadding="6">{lines}</table>"""

    class Registry(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            kind = "text/plain"
            student_port = urllib.parse.parse_qs(url.query).get("port", [""])[0]
            if url.path == "/register" and student_port.isdigit():
                ip = self.client_address[0]
                ip = me if ip == "127.0.0.1" else ip   # the instructor's own server
                body = servers[ip] = f"http://{ip}:{student_port}"
            elif url.path == "/list":                  # only the servers that answer now
                addresses = list(servers.values())
                body = "\n".join(a for a, (_, p) in zip(addresses, ask_all(addresses)) if p in ("0", "1"))
            else:
                body, kind = board(), "text/html"
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Registry)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'registry = "http://{me}:{port}"   board: http://localhost:{port}/')
    return server

if "class_board" in globals():                     # run again = restart (empty list)
    class_board.shutdown()
    class_board.server_close()
class_board = start_registry()
