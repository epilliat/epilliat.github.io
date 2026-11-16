# Class registry for A3d (instructor's laptop):  julia registry.jl
# Students register their server, then fetch the list of everyone's addresses.
#   GET /register?port=8011  -> records http://<caller's IP>:8011, answers it
#   GET /list                -> one address per line (one per machine)
using Sockets

PORT = 9000
servers = Dict{String,String}()   # caller IP -> its address (re-registering overwrites)

function handle(sock)
    path = split(readline(sock))[2]
    while !isempty(readline(sock)) end
    m = match(r"^/register\?port=(\d+)", path)
    if m === nothing
        body = join(values(servers), "\n")
    else
        ip = string(getpeername(sock)[1])
        body = servers[ip] = "http://$ip:$(m[1])"
        println(length(servers), " registered: ", body); flush(stdout)
    end
    write(sock, "HTTP/1.1 200 OK\r\nContent-Length: $(sizeof(body))\r\nConnection: close\r\n\r\n$body")
    close(sock)
end

server = listen(ip"0.0.0.0", PORT)
println("registry: http://$(getipaddr()):$PORT   (put this on the board)"); flush(stdout)
while true
    sock = accept(server)
    @async handle(sock)   # one task per request, all on one thread: no lock needed
end
