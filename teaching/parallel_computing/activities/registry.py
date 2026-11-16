# Class registry for A3d (instructor's laptop):  python3 registry.py
# Students register their server, then fetch the list of everyone's addresses.
#   GET /register?port=8011  -> records http://<caller's IP>:8011, answers it
#   GET /list                -> one address per line (one per machine)
import http.server, socket, threading, urllib.parse

PORT = 9000
servers = {}                      # caller IP -> its address (re-registering overwrites)
lock = threading.Lock()

class Registry(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/register":
            port = int(urllib.parse.parse_qs(url.query)["port"][0])
            address = f"http://{self.client_address[0]}:{port}"
            with lock:
                servers[self.client_address[0]] = address
            print(len(servers), "registered:", address, flush=True)
            body = address
        else:
            with lock:
                body = "\n".join(servers.values())
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass

def my_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))   # sends nothing
        return s.getsockname()[0]

print(f"registry: http://{my_address()}:{PORT}   (put this on the board)", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Registry).serve_forever()
