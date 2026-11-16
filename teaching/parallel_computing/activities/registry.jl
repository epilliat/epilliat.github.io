# Class registry for A3d (instructor's laptop):  julia registry.jl
# Students register their server, then fetch the list of everyone's addresses.
#   GET /register?port=8011  -> records http://<caller's IP>:8011, answers it
#   GET /list                -> one address per line (one per machine)
#   GET /                    -> page to project: names/messages, points, π (refreshes every 3 s)
using Sockets, Downloads

PORT = 9000
servers = Dict{String,String}()   # caller IP -> its address (re-registering overwrites)

function ask(url)
    try
        io = IOBuffer()
        Downloads.download(url, io; timeout=2)
        return String(take!(io))
    catch
        return nothing
    end
end

escape_html(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

function board()
    addresses = collect(values(servers))
    rows = Vector{Any}(undef, length(addresses))
    @sync for (k, a) in enumerate(addresses)
        @async rows[k] = (ask(a * "/name"), ask(a * "/inside"))
    end
    points = [inside for (_, inside) in rows if inside in ("0", "1")]
    hits = count(==("1"), points)
    title = isempty(points) ? "π ≈ ?" : "π ≈ $(round(4 * hits / length(points), digits=3))"
    lines = map(rows) do (name, inside)
        name = something(name, "")
        name = length(name) <= 40 ? name : first(name, 40) * "…"
        mark = inside == "1" ? "● inside" : inside == "0" ? "○ outside" : "no answer"
        "<tr><td>$mark</td><td>$(escape_html(name))</td></tr>"
    end
    """<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="3">
    <title>π by the class</title>
    <body style="font: 26px sans-serif; margin: 40px">
    <h1>$title <small>($(length(points)) points, $hits inside)</small></h1>
    <table cellpadding="6">$(join(lines))</table>"""
end

function handle(sock)
    path = split(readline(sock))[2]
    while !isempty(readline(sock)) end
    m = match(r"^/register\?port=(\d+)", path)
    kind = "text/plain"
    if m !== nothing
        ip = string(getpeername(sock)[1])
        body = servers[ip] = "http://$ip:$(m[1])"
        println(length(servers), " registered: ", body); flush(stdout)
    elseif path == "/list"
        body = join(values(servers), "\n")
    else
        body, kind = board(), "text/html"
    end
    write(sock, "HTTP/1.1 200 OK\r\nContent-Type: $kind; charset=utf-8\r\nContent-Length: $(sizeof(body))\r\nConnection: close\r\n\r\n$body")
    close(sock)
end

server = listen(ip"0.0.0.0", PORT)
println("registry: http://$(getipaddr()):$PORT   (put this on the board; open it to project)"); flush(stdout)
while true
    sock = accept(server)
    @async handle(sock)   # one task per request, all on one thread: no lock needed
end
