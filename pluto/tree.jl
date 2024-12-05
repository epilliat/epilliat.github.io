### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ ac6144d0-b171-11ef-096a-8ddf0a78717a
begin
	mutable struct BinaryTree{N,T}
		value::Union{
			Tuple{
				N,
				BinaryTree{N,T},
				BinaryTree{N,T}
			}, 
			T
		}
	end
end

# ╔═╡ 5636df9c-cb45-46bb-b08a-21898a8b2019
begin
	mutable struct Leave
		x::Real
		y::Real
	end
	mutable struct Node
		value::Real
	end
	const MyTree = BinaryTree{Node,Leave}
end

# ╔═╡ 435e2cc4-a842-4a66-af28-fd45380203a7
1 isa Real

# ╔═╡ a62ac808-df49-431e-8bc8-fa12f4383c1f
CvxTree(Leave(2))

# ╔═╡ f93da244-23f7-42e9-91c1-749491283875
b=MyTree((Node(2),a,a))

# ╔═╡ 05d1a0ab-948b-4c64-a7f1-a6284d2656ba
# ╠═╡ disabled = true
#=╠═╡
L= Leave(5)
  ╠═╡ =#

# ╔═╡ a503f152-e0f8-480e-80e2-4df2ea9ed406
#=╠═╡
CvxTree(L)
  ╠═╡ =#

# ╔═╡ 78bd07a7-cc17-4fb9-bc0b-c4c74b8e566f
#=╠═╡
Tree(L)
  ╠═╡ =#

# ╔═╡ 94347d14-6ff2-4fa9-ba4d-9797c82bba36
#=╠═╡
Tree(Tree(L))
  ╠═╡ =#

# ╔═╡ 68a6a5cb-1299-4b0c-94c0-d18d58bf1982
# ╠═╡ disabled = true
#=╠═╡
a = (l,1,1)
  ╠═╡ =#

# ╔═╡ 7fa63ff1-b1f3-4295-bb24-ae8b86ff1561
a=MyTree(Leave(2,3))

# ╔═╡ Cell order:
# ╠═ac6144d0-b171-11ef-096a-8ddf0a78717a
# ╠═5636df9c-cb45-46bb-b08a-21898a8b2019
# ╠═7fa63ff1-b1f3-4295-bb24-ae8b86ff1561
# ╠═f93da244-23f7-42e9-91c1-749491283875
# ╠═435e2cc4-a842-4a66-af28-fd45380203a7
# ╠═a503f152-e0f8-480e-80e2-4df2ea9ed406
# ╠═a62ac808-df49-431e-8bc8-fa12f4383c1f
# ╠═68a6a5cb-1299-4b0c-94c0-d18d58bf1982
# ╠═05d1a0ab-948b-4c64-a7f1-a6284d2656ba
# ╠═78bd07a7-cc17-4fb9-bc0b-c4c74b8e566f
# ╠═94347d14-6ff2-4fa9-ba4d-9797c82bba36
