### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a0000000-0000-4000-8000-000000000001
md"""
# Calcul parallèle — Multiple dispatch vs POO Python
## Comment Julia choisit la méthode à exécuter

**ENSAI 3A — Julia comme banc d'essai, Python comme point de comparaison**

---

On a vu *pourquoi* un même calcul peut être 100× plus rapide : compiler et **spécialiser par les types** supprime le surcoût par opération.

Ici, on regarde **le mécanisme du langage** qui rend cette spécialisation possible — le **multiple dispatch** — en le comparant frontalement à l'**orienté objet** (POO) que vous connaissez en Python.

> 🧭 Ce module est un **point de langage**, pas encore du parallélisme. Les threads viennent juste après — mais le dispatch est ce qui les rendra élégants (et ce qui, plus tard, fera basculer un calcul sur GPU sans changer le code).
"""

# ╔═╡ f3639896-4790-4121-ba28-654b4336fc4b
md"""
## 1. Le point de départ : la POO de Python

Vous connaissez l'**orienté objet** de Python : la méthode appartient à l'objet, et on l'appelle avec `objet.methode(...)`. Le choix de la méthode se fait sur **un seul** objet : celui à gauche du point.

Julia fait autrement : les fonctions sont **génériques**, et la version exécutée est choisie en fonction du **type de *tous* les arguments**. C'est le **multiple dispatch**.

Prenons le calcul de l'aire de formes géométriques.

### Version Python (orienté objet) — pour mémoire

```python
class Forme:
    def aire(self): ...

class Cercle(Forme):
    def __init__(self, r): self.r = r
    def aire(self): return 3.14159 * self.r**2

class Rectangle(Forme):
    def __init__(self, L, l): self.L, self.l = L, l
    def aire(self): return self.L * self.l

c = Cercle(2.0)
c.aire()      # le dispatch se fait sur "c"
```

La méthode `aire` est *enfermée dans la classe*. Pour ajouter une opération (disons `périmètre`), il faut **modifier chaque classe**.
"""

# ╔═╡ a0000000-0000-4000-8000-000000000002
md"""
## 2. La version Julia : multiple dispatch

En Julia, on sépare les **données** (les types) des **opérations** (les fonctions). D'abord les types…
"""

# ╔═╡ 3597c757-561c-47c6-9a37-72f42115be68
# Version Julia : on définit des types...
begin
    abstract type Forme end

    struct Cercle <: Forme
        r::Float64
    end

    struct Rectangle <: Forme
        L::Float64
        l::Float64
    end
    md"Types définis ✓"
end

# ╔═╡ 46ddf9ab-c365-46b4-bf65-2c2e4ca9551b
# ...puis une fonction générique "aire", avec une méthode par type
begin
    aire(c::Cercle)    = π * c.r^2
    aire(r::Rectangle) = r.L * r.l
end

# ╔═╡ 2d4a1901-4c31-4326-9715-1c7902b80ca0
let
    formes = [Cercle(2.0), Rectangle(3.0, 4.0), Cercle(1.0)]
    aire.(formes)        # le . applique aire à chaque élément
end

# ╔═╡ aa2ff25d-38c1-4909-b5b8-979cefd74299
md"""
## 3. La différence qui compte

En Python, `aire` vit **dans** l'objet. En Julia, `aire` est une fonction **extérieure** aux types, avec une méthode par combinaison de types.

Deux conséquences concrètes :

- **Extensibilité** : je peux ajouter une fonction `périmètre` pour `Cercle` et `Rectangle` **sans toucher** à leur définition. Je peux même ajouter un *nouveau* type `Triangle` et lui donner une méthode `aire` sans modifier le code existant. (En OO, ajouter une opération = modifier toutes les classes ; ajouter un type = facile. Le dispatch multiple rend les **deux** faciles — c'est le « problème d'expression ».)
- **Dispatch sur plusieurs arguments** : une fonction `collision(a, b)` peut avoir une méthode différente selon que `(a,b)` sont `(Cercle, Cercle)`, `(Cercle, Rectangle)`, etc. Impossible à exprimer naturellement avec `a.collision(b)`, qui ne regarde que `a`.
"""

# ╔═╡ b076a1ea-8d35-40ea-9d78-e58af1fe1d98
# Ajouter une opération SANS modifier les types : impossible aussi simplement en OO
begin
    périmètre(c::Cercle)    = 2π * c.r
    périmètre(r::Rectangle) = 2 * (r.L + r.l)
    périmètre.([Cercle(2.0), Rectangle(3.0, 4.0)])
end

# ╔═╡ d3fc1c50-e0a6-4409-9038-ac16231c0518
# Dispatch sur DEUX arguments : la méthode dépend du couple de types
begin
    interaction(a::Cercle, b::Cercle)       = "deux cercles"
    interaction(a::Cercle, b::Rectangle)    = "cercle et rectangle"
    interaction(a::Rectangle, b::Rectangle) = "deux rectangles"

    [interaction(Cercle(1.0), Cercle(2.0)),
     interaction(Cercle(1.0), Rectangle(2.0, 3.0)),
     interaction(Rectangle(1.0,1.0), Rectangle(2.0,2.0))]
end

# ╔═╡ 32ab7e99-08fc-488e-aff6-1641ec7f2110
md"""
## 4. Le lien avec la compilation

Ce n'est pas qu'une élégance de conception. **Le multiple dispatch est exactement ce qui permet la spécialisation par les types** vue dans le module précédent.

Quand vous appelez `aire(Cercle(2.0))`, Julia sait que l'argument est un `Cercle`, choisit la méthode correspondante, et **compile une version native spécialisée** pour ce type. C'est le même mécanisme que `f(3.0)` vs `f(3)` : un dispatch sur les types, suivi d'une compilation spécialisée.

Le point décisif est *quand* le type est connu. En Python (cf. le notebook Python), le dispatch se ferait **à l'exécution**, par recherche, à chaque appel, **sans** spécialisation. Ici les types sont **connus à la compilation** (JIT) : Julia choisit la méthode *et* génère le code natif une seule fois.

> **Distinction clé, côté dispatch :** type **découvert à l'exécution** (Python → lent) vs type **connu à la compilation** (Julia → spécialisé, rapide). Multiple dispatch + compilation JIT = des fonctions génériques **et** rapides, l'idée centrale du langage.

`methods(aire)` montre toutes les méthodes connues pour une fonction :
"""

# ╔═╡ 6ee8a88c-9ad2-48f3-81ff-b2c840e4254c
methods(aire)

# ╔═╡ a0000000-0000-4000-8000-000000000007
md"""
## 5. Bilan

| Idée | Python (POO) | Julia (dispatch) |
|---|---|---|
| Où vit la méthode | **dans** l'objet (`obj.methode()`) | **hors** des types, fonction générique |
| Choisie selon | le type de `obj` (un seul) | le type de **tous** les arguments |
| Ajouter un type | facile | facile |
| Ajouter une opération | modifier chaque classe | ajouter une fonction, sans toucher aux types |
| Lien avec la vitesse | — | **c'est le moteur** de la spécialisation/compilation |

> **À retenir.** Le dispatch multiple n'est pas qu'une question de style : c'est *le* mécanisme qui permet à Julia de générer du code natif spécialisé. On le reverra à l'œuvre quand un même `*` choisira tout seul la version **GPU** parce que ses arguments sont des `CuArray`.

### La suite

Place au **parallélisme sur CPU** : les **threads**, et un cas classique, la **somme parallèle qui donne un résultat faux**.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─a0000000-0000-4000-8000-000000000001
# ╟─f3639896-4790-4121-ba28-654b4336fc4b
# ╟─a0000000-0000-4000-8000-000000000002
# ╠═3597c757-561c-47c6-9a37-72f42115be68
# ╠═46ddf9ab-c365-46b4-bf65-2c2e4ca9551b
# ╠═2d4a1901-4c31-4326-9715-1c7902b80ca0
# ╟─aa2ff25d-38c1-4909-b5b8-979cefd74299
# ╠═b076a1ea-8d35-40ea-9d78-e58af1fe1d98
# ╠═d3fc1c50-e0a6-4409-9038-ac16231c0518
# ╟─32ab7e99-08fc-488e-aff6-1641ec7f2110
# ╠═6ee8a88c-9ad2-48f3-81ff-b2c840e4254c
# ╟─a0000000-0000-4000-8000-000000000007
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
