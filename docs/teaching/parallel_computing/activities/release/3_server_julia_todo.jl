# A slow server (Julia)  20 min.
# No Internet needed: this file starts its own web servers (0.2 s per answer). One thread.
# Run each #%% cell; run small timings twice (the 1st run compiles).

#%% 0. Setup (given): a slow server, and a function that fetches one URL
using Sockets, Downloads

function answer(sock, inside, name)
    path = split(readline(sock))[2]            # "GET /3 HTTP/1.1" -> "/3"
    while !isempty(readline(sock)) end         # skip the headers
    sleep(0.2)                                 # the server takes its time
    if path == "/inside"
        body = string(Int(inside))
    elseif path == "/name"
        body = name
    else
        body = path                            # otherwise: echoes the path
    end
    write(sock, "HTTP/1.1 200 OK\r\nContent-Length: $(sizeof(body))\r\nConnection: close\r\n\r\n$body")
    close(sock)
end

function start(one_per_request; host=ip"127.0.0.1", inside=false, name="")
    port, server = listenany(host, 8000)       # first free port from 8000
    @async while true
        sock = accept(server)
        if one_per_request
            @async answer(sock, inside, name)  # one task per request
        else
            answer(sock, inside, name)         # one request at a time
        end
    end
    return "http://127.0.0.1:$port"
end

function fetch_url(url)
    io = IOBuffer()
    Downloads.download(url, io; timeout=5)
    return String(take!(io))
end

concurrent = start(true)   # one task per request
serial = start(false)     # one request at a time

#%% 1. One request
@elapsed fetch_url(concurrent * "/1")

#%% 2. Two requests, one after the other
@elapsed begin
    fetch_url(concurrent * "/1")
    fetch_url(concurrent * "/2")
end

#%% 3. Two requests at once
@elapsed @sync begin
    @async fetch_url(concurrent * "/1")
    @async fetch_url(concurrent * "/2")
end

#%% 4. Same client code, the other server
@elapsed @sync begin
    @async fetch_url(serial * "/1")
    @async fetch_url(serial * "/2")
end

#%% 5. Now 20 requests with `workers` tasks: complete fetch_all
function fetch_all(urls, workers)
    out = Vector{String}(undef, length(urls))
    # ✏️ your turn: @sync over `workers` @async tasks; task w fetches urls w, w+workers, … into out
    error("remove this line and write your code")
    return out
end

urls = [concurrent * "/$i" for i in 1:20]
fetch_all(urls, 2) == ["/$i" for i in 1:20]   # right answers, in order

#%% 6. Play: change 5 to 1, 20…, and `concurrent` to `serial` in `urls`
@elapsed fetch_all(urls, 5)

#%% 7. The whole table
for server in (serial, concurrent), workers in (1, 5, 20)
    t = @elapsed fetch_all([server * "/$i" for i in 1:20], workers)
    println(server == concurrent ? "concurrent" : "serial", "  workers=", lpad(workers, 2), "  ", round(t, digits=2), " s")
end

#%% 8. Optional — π by the whole class: one random point each, your server answers it at /inside
x, y = rand(), rand()
inside = x^2 + y^2 <= 1                        # in the quarter disk?
my_name = ""                                       # optional: a name or a message
shared = start(true; host=ip"0.0.0.0", inside=inside, name=my_name)
registry = "http://127.0.0.1:9000"                 # replace by the address on the board
fetch_url(registry * "/register?port=" * split(shared, ":")[end])   # your address, as the others see it
# Nobody reaches the registry? School Wi-Fi may block it: try a phone hotspot.

#%% Everyone's points (Python or Julia servers), then π
addresses = split(fetch_url(registry * "/list"))
# ✏️ your turn: fetch "/inside" from every address at once with fetch_all, parse to Int, sum
error("remove this line and write your code")
4 * hits / length(addresses)

#%% Everyone's names or messages
addresses .=> fetch_all([a * "/name" for a in addresses], length(addresses))

#%% Instructor: the class board. Run it, then open http://localhost:9000/ (run by a student: a board of their own)
function start_registry(port=9000)
    servers = Dict{String,String}()            # IP -> address of its server
    me = string(getipaddr())
    function ask(url)                          # nothing if the server is gone
        try
            return fetch_url(url)
        catch
            return nothing
        end
    end
    function ask_all(addresses)                # (name, point) of every server, all at once
        rows = Vector{Any}(undef, length(addresses))
        @sync for (k, a) in enumerate(addresses)
            @async rows[k] = (something(ask(a * "/name"), ""), ask(a * "/inside"))
        end
        return rows
    end
    function board()
        rows = ask_all(collect(values(servers)))
        points = [p for (_, p) in rows if p in ("0", "1")]
        hits = count(==("1"), points)
        pi_hat = isempty(points) ? "?" : string(round(4 * hits / length(points), digits=3))
        safe(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
        cut(s) = length(s) <= 40 ? safe(s) : safe(first(s, 40)) * "&hellip;"
        mark = Dict("1" => "&#9679; inside", "0" => "&#9675; outside")
        lines = ["<tr><td>$(get(mark, p, "no answer"))</td><td>$(cut(n))</td></tr>" for (n, p) in rows]
        """<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="3">
        <body style="font: 26px sans-serif; margin: 40px"><p>registry = <b>"http://$me:$port"</b></p>
        <h1>&pi; &asymp; $pi_hat <small>($(length(points)) points, $hits inside)</small></h1>
        <table cellpadding="6">$(join(lines))</table>"""
    end
    function handle(sock)
        path = split(readline(sock))[2]
        while !isempty(readline(sock)) end
        m = match(r"^/register\?port=(\d+)", path)
        kind = "text/plain"
        if m !== nothing
            ip = string(getpeername(sock)[1])
            ip = ip == "127.0.0.1" ? me : ip       # the instructor's own server
            body = servers[ip] = "http://$ip:$(m[1])"
        elseif path == "/list"                 # only the servers that answer now
            addresses = collect(values(servers))
            body = join([a for (a, (_, p)) in zip(addresses, ask_all(addresses)) if p in ("0", "1")], "\n")
        else
            body, kind = board(), "text/html"
        end
        write(sock, "HTTP/1.1 200 OK\r\nContent-Type: $kind; charset=utf-8\r\nContent-Length: $(sizeof(body))\r\nConnection: close\r\n\r\n$body")
        close(sock)
    end
    server = listen(ip"0.0.0.0", port)
    @async try
        while true
            sock = accept(server)
            @async handle(sock)
        end
    catch                                      # closed: restarted
    end
    println("registry = \"http://$me:$port\"   board: http://localhost:$port/")
    return server
end

@isdefined(class_board) && close(class_board)  # run again = restart (empty list)
class_board = start_registry()
