# Class registry for A3d (instructor's laptop):  python3 registry.py
# Students register their server, then fetch the list of everyone's addresses.
#   GET /register?port=8011  -> records http://<caller's IP>:8011, answers it
#   GET /list                -> one address per line (one per machine)
#   GET /                    -> page to project: names/messages, points, π (refreshes every 3 s)
import html, http.server, socket, threading, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = 9000
servers = {}                      # caller IP -> its address (re-registering overwrites)
lock = threading.Lock()

def ask(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.read().decode()
    except OSError:
        return None

def row(address):                 # (name, inside) of one student; None if no answer
    return ask(address + "/name"), ask(address + "/inside")

def board():
    with lock:
        addresses = list(servers.values())
    with ThreadPoolExecutor(max_workers=max(1, len(addresses))) as pool:
        rows = list(pool.map(row, addresses))
    points = [inside for _, inside in rows if inside in ("0", "1")]
    hits = sum(int(p) for p in points)
    title = f"π ≈ {4 * hits / len(points):.3f}" if points else "π ≈ ?"
    lines = []
    for name, inside in rows:
        name = name or ""
        name = name if len(name) <= 40 else name[:40] + "…"
        mark = {"1": "● inside", "0": "○ outside"}.get(inside, "no answer")
        lines.append(f"<tr><td>{mark}</td><td>{html.escape(name)}</td></tr>")
    return f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<title>π by the class</title>
<body style="font: 26px sans-serif; margin: 40px">
<h1>{title} <small>({len(points)} points, {hits} inside)</small></h1>
<table cellpadding="6">{''.join(lines)}</table>"""

class Registry(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        kind = "text/plain"
        if url.path == "/register":
            port = int(urllib.parse.parse_qs(url.query)["port"][0])
            address = f"http://{self.client_address[0]}:{port}"
            with lock:
                servers[self.client_address[0]] = address
            print(len(servers), "registered:", address, flush=True)
            body = address
        elif url.path == "/list":
            with lock:
                body = "\n".join(servers.values())
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

def my_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))   # sends nothing
        return s.getsockname()[0]

print(f"registry: http://{my_address()}:{PORT}   (put this on the board; open it to project)", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Registry).serve_forever()
