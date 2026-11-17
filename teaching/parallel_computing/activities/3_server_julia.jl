# A slow server (Julia)  20 min.
# No Internet needed: this file starts its own web servers (0.2 s per answer). One thread.
# Run each #%% cell; run small timings twice (the 1st run compiles).

#%% 0. Setup (given): a slow server, and a function that fetches one URL
using Sockets, Downloads

function answer(sock, inside, name)
    path = split(readline(sock))[2]            # "GET /3 HTTP/1.1" -> "/3"
    while !isempty(readline(sock)) end         # skip the headers
    sleep(0.2)                                 # the server takes its time
    body = path == "/inside" ? string(Int(inside)) :
           path == "/name"   ? name : path      # otherwise: echoes the path
    write(sock, "HTTP/1.1 200 OK\r\nContent-Length: $(sizeof(body))\r\nConnection: close\r\n\r\n$body")
    close(sock)
end

function start(one_per_request; host=ip"127.0.0.1", inside=false, name="")
    port, server = listenany(host, 8000)       # first free port from 8000
    @async while true
        sock = accept(server)
        one_per_request ? (@async answer(sock, inside, name)) : answer(sock, inside, name)
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

fetch_url(concurrent * "/inside")    # → "0" ici, ou inside pour le serveur `shared`

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
    #= SOLUTION: @sync over `workers` @async tasks; task w fetches urls w, w+workers, … into out =#
    @sync for w in 1:workers
        @async for k in w:workers:length(urls)
            out[k] = fetch_url(urls[k])
        end
    end
    #= END =#
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
my_name = ""                                   # optional: a name or a message
shared = start(true; host=ip"0.0.0.0", inside=inside, name=my_name)
registry = "http://127.0.0.1:9000"  
registry * "/register?port=" * split(shared, ":")[end]           # replace by the address on the board
fetch_url(registry * "/register?port=" * split(shared, ":")[end])   # your address, as the others see it
# Nobody reaches the registry? School Wi-Fi may block it: try a phone hotspot.

#%% Everyone's points (Python or Julia servers), then π
addresses = split(fetch_url(registry * "/list"))
#= SOLUTION: fetch "/inside" from every address at once with fetch_all, parse to Int, sum =#
hits = sum(parse.(Int, fetch_all([a * "/inside" for a in addresses], length(addresses))))
#= END =#
4 * hits / length(addresses)

#%% Everyone's names or messages
addresses .=> fetch_all([a * "/name" for a in addresses], length(addresses))
