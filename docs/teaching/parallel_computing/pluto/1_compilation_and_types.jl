### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ab181e11-6516-4169-9189-fb4488f4c8d3
md"""
# Interpreted vs compiled
## Why the same computation can be 100× faster

**ENSAI 3A — Julia as a test bench, Python as a point of comparison.** For each cell: **predict, then run.**

Plan: the JIT warmup → measuring properly → type specialization → *(bonus)* SIMD → Monte-Carlo π.
"""

# ╔═╡ e4c6629b-29ce-4e63-b529-04bf1020ee4f
md"""
## 0. Getting started

The first run downloads and precompiles the packages — already an illustration: Julia compiles.
"""

# ╔═╡ 86593a2f-41c2-464d-905a-70b368785d81
begin
    using BenchmarkTools   # @btime, @benchmark
    using PlutoUI          # slider (interactive bonus)
    using Random           # MersenneTwister (reproducible darts)
    md"Packages loaded ✓"
end

# ╔═╡ 7525bf04-f5c7-45aa-b826-642c15b68eaf
@btime 1 + 2

# ╔═╡ 9c279c3b-f2c3-4a98-8f21-b21190182e5b
md"""
`@btime 1 + 2` says **~1 ns**… yet the cell took seconds. **Why?**

## 1. The JIT warmup — a hand-written loop on 10M floats
"""

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81176
N = 10_000_000

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81177
v = rand(N)

# ╔═╡ 26289681-0477-4670-950f-8351be0fdd5a
function mysum(x)
    s = zero(eltype(x))    # a zero of the element type (0.0 here)
    for i in eachindex(x)  # the valid indices of x
        s += x[i]
    end
    return s
end

# ╔═╡ da96898d-efc5-4e47-a73e-8e0a991bc0f8
mysum(v)

# ╔═╡ bb0ace77-07a4-4504-80c1-edc793f3f899
md"""
The same loop in pure Python (`for xi in x: s += xi`) takes **~0.5 s**. **Bet** on Julia's time — we measure it in §2.
"""

# ╔═╡ 05891168-bf4c-42c6-86f1-5d038606e267
md"""
**The warmup over 10 calls:** which call is slow, and why only that one?
"""

# ╔═╡ b1180cde-dd1e-4983-aae0-bd631034c0df
let                                 # let … end = a local scope
    function fresh(x)
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]
        end
        return s
    end
    fb = Function[fresh]            # hides the type: compilation lands in call 1
    times = [@elapsed(fb[1](v)) for _ in 1:10]
    println("ms per call: ", round.(times .* 1e3, digits=3))
end

# ╔═╡ 97acf5d7-5b90-4b3a-a2b9-ffdda2d75a52
md"""
Full experiment (5 variants, 2000 calls): `experiences/warmup_jit/`. The numpy side (fast inside, slow in a Python loop): `python/1_compilation_and_types_python.ipynb`.
"""

# ╔═╡ d02cd083-dea1-417f-a941-8eac616197a6
md"""
## 2. Measuring properly — compare the two timings
"""

# ╔═╡ a8128f4d-9cdc-4cd2-8025-33c5935757e0
let
    g(x) = sum(abs2, x)
    @time g(v)
    @time g(v)
end

# ╔═╡ 46c146f1-d1e4-4797-87bc-2c03a0185daf
md"""
**Read the distribution:** why is the *minimum* the number to quote?
"""

# ╔═╡ 3533f187-1da1-452b-9a63-0ea1bd69b4f6
@btime mysum($v)

# ╔═╡ ca49af93-0baa-47ee-a4ec-3849a1cacffb
@benchmark mysum($v)

# ╔═╡ 90e32a56-6c3c-4248-86d3-74e504341707
md"""
Then try `@btime mysum(v)` **without `$`**: what changes? *(The why — untyped globals — comes with dispatch.)*
"""

# ╔═╡ 22d3648b-a71a-4299-8e1b-46f032a74046
md"""
## 3. Type specialization — step 1, lowered code

The warmup spike was Julia **compiling**. Let's look at each stage, on a deliberately trivial function.
"""

# ╔═╡ a51ea6fa-9f15-453f-bcea-852c7e71a354
f(x) = 2x + 1

# ╔═╡ 26118708-e6fa-4cdb-a5e9-7a72a376b84f
@code_lowered f(3.0)

# ╔═╡ c3c88228-8ec2-4a08-9384-7da546be08fb
md"""
### Step 2 — typed code: compare `Float64` and `Int`
"""

# ╔═╡ c14f0bba-a2a2-457d-a400-35224ec23b34
@code_typed f(3.0)

# ╔═╡ eb5c0a66-0e2e-4eba-9cc6-0b6f6acd4fb6
@code_typed f(3)

# ╔═╡ 3e4c87e0-e1aa-4593-8f71-a57b44ef14fe
md"""
### Step 3 — assembly: how many instructions? Float or integer ones?
"""

# ╔═╡ cf7f8128-23b8-4c2e-a122-d2705f2c43c6
@code_native debuginfo=:none f(3.0)

# ╔═╡ d5000000-0000-4d00-9b00-00000000d001
@code_native debuginfo=:none f(3)

# ╔═╡ cb125025-5a35-4e36-bf13-7efc3e29abd4
md"""
One function, **one compiled version per argument type**, a few machine instructions each, no interpreter left. CPython ≥ 3.11 also specializes — but one bytecode at a time, re-checking the type guard at every operation, on boxed objects.
"""

# ╔═╡ 656f12ed-0f30-498d-8462-f58bff519eda
md"""
## 4. Bonus (optional) — Base `sum` vs our loop

A vector that fits in cache, to isolate the computation from memory:
"""

# ╔═╡ 6f3bc084-05bc-4b17-98f6-5d306b03740e
vc = rand(100_000)

# ╔═╡ c3c32bc0-9d67-46e2-bdbe-c7d3f84ee2f8
@btime mysum($vc)

# ╔═╡ 42fbb9f9-240e-487c-8f3d-ad839115f426
@btime sum($vc)

# ╔═╡ e8b6b134-eb4f-49f2-8f39-eb3c3f1365e9
md"""
Our loop's assembly: `vaddsd` = *scalar double*. How many accumulators?
"""

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82804
@code_native debuginfo=:none mysum(vc)

# ╔═╡ ba010064-49f9-4872-a4fe-a42d2e754d75
md"""
`sum`'s assembly: look for `vaddpd`… then for the `call`.
"""

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82805
@code_native debuginfo=:none sum(vc)

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82806
md"""
The kernel `sum` calls: what replaces `vaddsd`? How many accumulators?
"""

# ╔═╡ 9301ea0f-f270-4015-8acd-f23507b82807
@code_native debuginfo=:none Base.mapreduce_impl(identity, Base.add_sum, vc, 1, length(vc), 1024)

# ╔═╡ ba010064-49f9-4872-a4fe-a42d2e754d76
md"""
Why can't the compiler do this to **our** loop? Hint: is `(a+b)+c == a+(b+c)` in floats?

### `@simd`: permission to reassociate
"""

# ╔═╡ 8e0cdaff-257b-4836-8998-2cd5f0ef8044
function mysum_simd(x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)   # @inbounds: skip bounds checks
        s += x[i]
    end
    return s
end

# ╔═╡ 0662912c-e7ab-4223-82d2-e2683543f99b
@btime mysum_simd($vc)

# ╔═╡ 49ea0717-5e4a-44df-a895-11b9d55c747f
md"""
## 5. Monte-Carlo π — same count as in the Julia-basics module

$$\pi \approx 4 \times \frac{\#\{x^2+y^2 \le 1\}}{n}$$
"""

# ╔═╡ e342c5aa-9ddc-46d3-8ac2-c6f90e1b855a
function count_in_disk(n)
    hits = 0
    for _ in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 ≤ 1
            hits += 1
        end
    end
    return hits
end

# ╔═╡ 760cd379-4178-4e7f-9695-a1d709b36969
estimate_pi(n) = 4 * count_in_disk(n) / n

# ╔═╡ 8adcfe41-1d0d-475a-968c-5a5a971a142b
estimate_pi(10_000_000)

# ╔═╡ 855d77ed-ca4b-407a-b225-2be1a2c499f5
md"""
**Q1** — how long? How many allocations?
"""

# ╔═╡ 3cb6763c-dc00-4c09-94b5-289f773dc460
@btime estimate_pi(10_000_000)

# ╔═╡ f15bd89d-47f6-4c87-a3d9-17078b722f1e
md"""
**Q2** — type-stable? Is the yellow `Union{Nothing, Tuple…}` a problem? Why always `Float64`?

`@code_warntype` = `@code_typed` with uncertain types in color (red `Any` = unknown).
"""

# ╔═╡ 654b39b4-b592-4066-b519-0783a704f637
@code_warntype count_in_disk(1000)

# ╔═╡ 654b39b4-b592-4066-b519-0783a704f638
Base.return_types(estimate_pi, (Int,))

# ╔═╡ 49546ce1-11ee-4cc8-b412-24b9e933a728
md"""
**Q3** — how does the error shrink with `n`? Why is this ideal for parallelism?
"""

# ╔═╡ b7792bef-9262-42b4-98d0-407113ded36c
let
    for n in (10^4, 10^5, 10^6, 10^7, 10^8)
        err = abs(estimate_pi(n) - π)
        println("n = 10^$(round(Int, log10(n)))   error = $(round(err, digits=5))")
    end
end

# ╔═╡ 5973d547-5a91-4d4b-8696-88dfbf03a0a3
md"""
## Bonus (Pluto only) — watch π converge

**Green** darts land in the quarter disk, **red** outside; the estimate is 4 × green / n. Drag the slider: does it settle like 1/√n (Q3)?
"""

# ╔═╡ c687db86-e0a1-4083-ba96-d5a9af91738c
md"""**Number of darts:** $(@bind n_pts Slider(100:100:3000, default=800, show_value=true))"""

# ╔═╡ 7f18d64c-dc08-4d31-850a-a7e3bfc4bdcf
let
    S = 360
    rng = MersenneTwister(1)
    pts = [(rand(rng), rand(rng)) for _ in 1:n_pts]
    inside(p) = p[1]^2 + p[2]^2 <= 1.0
    nin = count(inside, pts)
    est = 4 * nin / n_pts
    Xp(x) = round(x * S, digits=1)
    Yp(y) = round(S * (1 - y), digits=1)
    arc = join(("$(Xp(cos(t))),$(Yp(sin(t)))" for t in range(0, pi/2, length=80)), " ")
    dots = join(("<circle cx='$(Xp(p[1]))' cy='$(Yp(p[2]))' r='2.5' fill='$(inside(p) ? "#2a9d8f" : "#e76f51")' fill-opacity='0.75'/>" for p in pts))
    HTML("""
    <div style="font-family:sans-serif">
      <svg width="$S" height="$S" viewBox="0 0 $S $S" style="background:#fff;border:1px solid #bbb">
        <polygon points="$(Xp(0)),$(Yp(0)) $arc" fill="#a8dadc" fill-opacity="0.3"/>
        <polyline points="$arc" fill="none" stroke="#1d3557" stroke-width="2.5"/>
        <rect x="0.5" y="0.5" width="$(S-1)" height="$(S-1)" fill="none" stroke="#333" stroke-width="1.5"/>
        $dots
      </svg>
      <div style="margin-top:6px;font-size:17px">
        <b>π ≈ $(round(est, digits=4))</b> · n = $n_pts · inside = $nin · error = $(round(abs(est - pi), digits=4))
      </div>
    </div>
    """)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "c23e975c6e744836faae74be2696ce7fec871174"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "468dbe2b510c876dc091b2c74ed52c7c34f48b9b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"
"""

# ╔═╡ Cell order:
# ╟─ab181e11-6516-4169-9189-fb4488f4c8d3
# ╟─e4c6629b-29ce-4e63-b529-04bf1020ee4f
# ╠═86593a2f-41c2-464d-905a-70b368785d81
# ╠═7525bf04-f5c7-45aa-b826-642c15b68eaf
# ╟─9c279c3b-f2c3-4a98-8f21-b21190182e5b
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81176
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81177
# ╠═26289681-0477-4670-950f-8351be0fdd5a
# ╠═da96898d-efc5-4e47-a73e-8e0a991bc0f8
# ╟─bb0ace77-07a4-4504-80c1-edc793f3f899
# ╟─05891168-bf4c-42c6-86f1-5d038606e267
# ╠═b1180cde-dd1e-4983-aae0-bd631034c0df
# ╟─97acf5d7-5b90-4b3a-a2b9-ffdda2d75a52
# ╟─d02cd083-dea1-417f-a941-8eac616197a6
# ╠═a8128f4d-9cdc-4cd2-8025-33c5935757e0
# ╟─46c146f1-d1e4-4797-87bc-2c03a0185daf
# ╠═3533f187-1da1-452b-9a63-0ea1bd69b4f6
# ╠═ca49af93-0baa-47ee-a4ec-3849a1cacffb
# ╟─90e32a56-6c3c-4248-86d3-74e504341707
# ╟─22d3648b-a71a-4299-8e1b-46f032a74046
# ╠═a51ea6fa-9f15-453f-bcea-852c7e71a354
# ╠═26118708-e6fa-4cdb-a5e9-7a72a376b84f
# ╟─c3c88228-8ec2-4a08-9384-7da546be08fb
# ╠═c14f0bba-a2a2-457d-a400-35224ec23b34
# ╠═eb5c0a66-0e2e-4eba-9cc6-0b6f6acd4fb6
# ╟─3e4c87e0-e1aa-4593-8f71-a57b44ef14fe
# ╠═cf7f8128-23b8-4c2e-a122-d2705f2c43c6
# ╠═d5000000-0000-4d00-9b00-00000000d001
# ╟─cb125025-5a35-4e36-bf13-7efc3e29abd4
# ╟─656f12ed-0f30-498d-8462-f58bff519eda
# ╠═6f3bc084-05bc-4b17-98f6-5d306b03740e
# ╠═c3c32bc0-9d67-46e2-bdbe-c7d3f84ee2f8
# ╠═42fbb9f9-240e-487c-8f3d-ad839115f426
# ╟─e8b6b134-eb4f-49f2-8f39-eb3c3f1365e9
# ╠═9301ea0f-f270-4015-8acd-f23507b82804
# ╟─ba010064-49f9-4872-a4fe-a42d2e754d75
# ╠═9301ea0f-f270-4015-8acd-f23507b82805
# ╟─9301ea0f-f270-4015-8acd-f23507b82806
# ╠═9301ea0f-f270-4015-8acd-f23507b82807
# ╟─ba010064-49f9-4872-a4fe-a42d2e754d76
# ╠═8e0cdaff-257b-4836-8998-2cd5f0ef8044
# ╠═0662912c-e7ab-4223-82d2-e2683543f99b
# ╟─49ea0717-5e4a-44df-a895-11b9d55c747f
# ╠═e342c5aa-9ddc-46d3-8ac2-c6f90e1b855a
# ╠═760cd379-4178-4e7f-9695-a1d709b36969
# ╠═8adcfe41-1d0d-475a-968c-5a5a971a142b
# ╟─855d77ed-ca4b-407a-b225-2be1a2c499f5
# ╠═3cb6763c-dc00-4c09-94b5-289f773dc460
# ╟─f15bd89d-47f6-4c87-a3d9-17078b722f1e
# ╠═654b39b4-b592-4066-b519-0783a704f637
# ╠═654b39b4-b592-4066-b519-0783a704f638
# ╟─49546ce1-11ee-4cc8-b412-24b9e933a728
# ╠═b7792bef-9262-42b4-98d0-407113ded36c
# ╟─5973d547-5a91-4d4b-8696-88dfbf03a0a3
# ╟─c687db86-e0a1-4083-ba96-d5a9af91738c
# ╟─7f18d64c-dc08-4d31-850a-a7e3bfc4bdcf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
