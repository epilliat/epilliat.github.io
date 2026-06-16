### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 79d9e4bb-4e4b-4d59-83a2-7b03d73c6b0d
md"""
# Calcul parallèle — Threads et concurrence
## Faire travailler plusieurs cœurs… sans corrompre le résultat

**ENSAI 3A — Julia comme banc d'essai, Python comme point de comparaison**

---

On sait *pourquoi* Julia peut être rapide (compiler et **spécialiser par les types**) et *comment* il choisit ses méthodes (le **multiple dispatch**). Place maintenant au **parallélisme sur CPU** : faire travailler plusieurs cœurs en même temps.

Et surtout : **le piège de la somme parallèle qui donne un résultat faux**, qu'on va déclencher de nos propres mains.

> ⚙️ **Avant de commencer** : ce notebook a besoin de **plusieurs threads**. Vérifiez la cellule 0 ci-dessous. Si elle affiche `1`, relancez Julia avec plusieurs threads (voir l'instruction).
"""

# ╔═╡ 43328603-10db-498a-bd37-3f7167c76c97
md"""
## 0. Combien de threads avons-nous ?
"""

# ╔═╡ 36881278-4c1e-430d-8e68-bf043782b8a3
begin
    using BenchmarkTools
    using PlutoUI
    using Base.Threads
    using Printf
    md"Paquets chargés ✓"
end

# ╔═╡ 6186c4ef-f768-4556-ba3c-43ff5975d5ea
Threads.nthreads()

# ╔═╡ 367ad1ec-8ac8-4e9b-99d9-9aafd2fa7fa9
md"""
**Si la cellule ci-dessus affiche `1`**, Julia n'a qu'un seul thread et les démos de parallélisme ne montreront aucune accélération.

Pour donner plusieurs threads à Julia, lancez-le ainsi **avant** d'ouvrir Pluto :

```bash
julia --threads=auto        # autant de threads que de cœurs
# ou un nombre précis :
julia --threads=4
```

puis dans cette session Julia :

```julia
import Pluto; Pluto.run()
```

Dans Pluto récent, vous pouvez aussi fixer le nombre de threads via le menu. Vérifiez que la cellule affiche bien `> 1` avant de continuer.
"""

# ╔═╡ 25b0c9b5-0a5a-450a-b686-8b5cb2bb39e4
md"""
## 1. Pourquoi le threading Python ne parallélise pas le CPU

Avant de paralléliser, un mot sur Python — parce que ça explique une bonne partie de l'écosystème.

CPython a un **GIL** (*Global Interpreter Lock*) : un verrou global qui fait qu'**un seul thread Python exécute du bytecode à la fois**. Conséquence :

- Le module `threading` de Python **ne** donne **aucune** accélération sur un calcul CPU pur : les threads se relaient mais ne s'exécutent jamais vraiment en même temps.
- Pour utiliser plusieurs cœurs en Python, on passe par `multiprocessing` (plusieurs *processus*, donc plusieurs interpréteurs, donc de la copie de données et de la communication coûteuse), ou on sort vers du C/numpy/Numba qui libère le GIL.

Julia n'a pas de GIL : `Threads.@threads` exécute **réellement** en parallèle sur plusieurs cœurs, en **mémoire partagée** (tous les threads voient les mêmes tableaux). C'est plus puissant… et c'est précisément ce qui ouvre la porte aux **bugs de concurrence** qu'on va voir maintenant.

> 🔎 Note : Python évolue (un mode « sans GIL » expérimental existe dans les versions récentes), mais pour l'écrasante majorité du code Python en production aujourd'hui, le raisonnement ci-dessus tient.
"""

# ╔═╡ 27632886-8768-498a-8544-8a12dd300b6c
md"""
## 2. Un premier cas : une boucle parallèle correcte

Commençons par un cas **sans piège** : appliquer une fonction coûteuse à chaque élément d'un tableau, en écrivant **chaque résultat à un endroit différent**. Comme les threads n'écrivent jamais au même endroit, tout va bien.
"""

# ╔═╡ 4c5addbf-2a8f-455d-9364-2a8cbc60e8c6
# une fonction volontairement un peu coûteuse
function travail(x)
    s = 0.0
    for k in 1:200
        s += sin(x + k) * cos(x - k)
    end
    return s
end

# ╔═╡ af9c646c-9738-4ee0-8afa-4275f91666f9
# version séquentielle : chaque case du résultat calculée l'une après l'autre
function map_seq(xs)
    out = similar(xs)
    for i in eachindex(xs)
        out[i] = travail(xs[i])
    end
    return out
end

# ╔═╡ a6417f05-1d4a-41c0-96bb-2dd9d7f66178
# version parallèle : @threads répartit les indices i entre les threads.
# Chaque thread écrit dans out[i] — des cases DIFFÉRENTES. Aucun conflit.
function map_par(xs)
    out = similar(xs)
    @threads for i in eachindex(xs)
        out[i] = travail(xs[i])
    end
    return out
end

# ╔═╡ 00f33fed-d611-4d0a-b6e8-b6e121331667
begin
    const données = rand(2_000_000)
    map_seq(données) == map_par(données)   # même résultat : bon signe
end

# ╔═╡ 46f8d94c-59de-4334-9ce7-3df3c9ef027e
let
    println("Threads disponibles : ", Threads.nthreads())
    print("Séquentiel : "); @btime map_seq($données)
    print("Parallèle  : "); @btime map_par($données)
end

# ╔═╡ 37a835de-b514-429a-a71c-fee0ed868de5
md"""
**Discussion.** Le facteur d'accélération sera proche du nombre de threads… **mais jamais exactement** (loi d'Amdahl : la partie non parallélisable — allocation, répartition — plafonne le gain). On y reviendra avec le GPU, où ce raisonnement devient critique.

Ce cas marche parce que **chaque thread écrit dans sa propre case**. Le danger, c'est quand **plusieurs threads écrivent au même endroit**.
"""

# ╔═╡ e5eca89b-c38c-46c2-8355-801b6d1a1987
md"""
## 3. La somme parallèle fausse (cas classique)

On veut sommer un grand vecteur, en parallèle. Idée « naïve » : un accumulateur partagé `s`, et chaque thread ajoute sa contribution.

**Cette cellule contient un bug.** On va le déclencher et le constater.
"""

# ╔═╡ c683816e-ae0b-4c41-b730-14a84bac69df
function somme_fausse(xs)
    s = 0.0
    @threads for i in eachindex(xs)
        s += xs[i]          # ⚠️ tous les threads écrivent dans le MÊME s
    end
    return s
end

# ╔═╡ 5268900e-a75e-4651-9692-1fcefe334e09
begin
    const w = rand(10_000_000)
    const vraie_somme = sum(w)        # la référence, calculée par Julia
    vraie_somme
end

# ╔═╡ 2bb190de-2f19-41a9-bb35-0122f4440cb1
md"""
Exécutons `somme_fausse` **plusieurs fois de suite**. Si le calcul était correct, on obtiendrait toujours le même nombre. Regardez bien.
"""

# ╔═╡ 3662368f-7669-439b-a879-dfa8bd39e327
let
    println("Vraie somme : ", vraie_somme, "\n")
    for essai in 1:8
        r = somme_fausse(w)
        err = abs(r - vraie_somme)
        println(@sprintf("essai %d : %.4f   (erreur %.4f)", essai, r, err))
    end
end

# ╔═╡ c3a7201f-dec2-4031-be82-b71e3e30bf0b
md"""
### Que s'est-il passé ?

Deux symptômes, tous les deux mauvais :

1. **Le résultat est faux** (trop petit).
2. **Il change à chaque exécution** : le calcul est devenu **non déterministe**.

La cause est une **situation de compétition** (*race condition*). L'opération `s += xs[i]` n'est **pas atomique** : elle se décompose en trois temps —

- **lire** la valeur actuelle de `s` dans un registre ;
- **ajouter** `xs[i]` ;
- **réécrire** le résultat dans `s`.

Quand deux threads font ça « en même temps », ils lisent tous les deux la même ancienne valeur de `s`, ajoutent chacun *leur* élément, et le second à réécrire **écrase** la contribution du premier. Des additions sont **perdues** — et combien, ça dépend du hasard de l'ordonnancement. D'où un résultat trop petit et différent à chaque fois.

> C'est **la** leçon de ce module : en mémoire partagée, *écrire au même endroit depuis plusieurs threads sans précaution = corruption silencieuse*. Pas de message d'erreur. Juste un résultat faux.
"""

# ╔═╡ c5c4b1ab-fd6f-407c-b4cd-df77249d85f2
md"""
## 4. Trois façons de réparer

### Réparation A — accumulateurs locaux + réduction (la bonne approche)

Chaque thread accumule dans **sa propre** variable (pas de partage, donc pas de conflit), puis on combine les totaux partiels **à la fin**. C'est le patron *map-reduce*, et c'est celui à retenir.
"""

# ╔═╡ 02945be2-9202-4973-b755-ae827fc03dff
function somme_locale(xs)
    # un accumulateur par thread
    partiels = zeros(Float64, Threads.nthreads())
    @threads for i in eachindex(xs)
        t = Threads.threadid()
        partiels[t] += xs[i]      # chaque thread n'écrit QUE dans SA case
    end
    return sum(partiels)          # réduction finale, séquentielle et rapide
end

# ╔═╡ 7044c222-4a99-484d-ad76-29bc5ed056d7
let
    println("Vraie somme  : ", vraie_somme)
    for essai in 1:4
        println(@sprintf("somme_locale : %.6f", somme_locale(w)))
    end
end

# ╔═╡ f9287ed8-08f8-4681-87d9-6e25c117e9b6
md"""
Correct **et** reproductible. Chaque thread n'écrit que dans `partiels[threadid()]` — une case qui lui est réservée. Aucune compétition.

> ⚠️ Subtilité avancée (à mentionner, pas à détailler) : cette version par `threadid()` peut souffrir de *false sharing* — deux cases voisines de `partiels` tombant sur la même ligne de cache, ce qui ralentit sans fausser. On en parle dans le module sur la mémoire. La version idéale utilise une variable vraiment locale à chaque tâche.
"""

# ╔═╡ 18174e65-4a39-48cc-ab55-289da50e0f67
md"""
### Réparation B — une opération atomique

On peut demander au matériel de rendre l'addition **indivisible** : `@atomic` garantit que lire-ajouter-réécrire se fait d'un seul bloc. Correct… mais **lent** si on l'utilise à chaque élément : tous les threads se sérialisent sur le même verrou.
"""

# ╔═╡ c0d3dfab-8f26-4359-9196-a99c76f1553d
function somme_atomique(xs)
    s = Threads.Atomic{Float64}(0.0)
    @threads for i in eachindex(xs)
        Threads.atomic_add!(s, xs[i])   # indivisible, mais sérialisé (les threads attendent)
    end
    return s[]
end

# ╔═╡ e027d2e9-230e-418b-81b1-8c78e3314048
let
    println("Vraie somme    : ", vraie_somme)
    println("somme_atomique : ", somme_atomique(w))
end

# ╔═╡ a990541e-c4f7-4e90-971c-7dbb0bb224c8
md"""
### Comparons les trois sur le temps

Le verdict est instructif : la version atomique est **correcte mais souvent plus lente que le séquentiel**, parce que les threads passent leur temps à s'attendre. La version à accumulateurs locaux est correcte **et** rapide.
"""

# ╔═╡ 98c2ad45-47d3-41f5-b077-1dc0b8d3f3bf
let
    print("séquentiel (sum) : "); @btime sum($w)
    print("locale           : "); @btime somme_locale($w)
    print("atomique         : "); @btime somme_atomique($w)
    # on ne benchmarke pas somme_fausse : elle est fausse, son temps n'a pas de sens
end

# ╔═╡ 191e9b94-e2a1-49dd-a689-30754d41a247
md"""
### La morale

| Approche | Correct ? | Rapide ? | Pourquoi |
|---|---|---|---|
| `s += ...` partagé | ❌ | — | race condition, additions perdues |
| `@atomic` partout | ✅ | ❌ | sérialisation sur le verrou |
| accumulateurs locaux + réduction | ✅ | ✅ | aucun partage dans la boucle chaude |

> **Paralléliser une réduction, ce n'est pas « ajouter `@threads` devant la boucle ».** C'est repenser l'algorithme pour éviter les accès concurrents au même emplacement mémoire. Ce principe — *minimiser le partage* — sera encore plus déterminant sur GPU.
"""

# ╔═╡ c77e2fc7-56e1-4c64-9776-899fd97d661e
md"""
## 5. Activité — paralléliser le Monte-Carlo de π

On reprend l'exemple conducteur du cours (le Monte-Carlo de π). Voici la version séquentielle.
"""

# ╔═╡ 6ca5e3d9-ebf5-45ff-9886-ab465e3e51e1
function pi_seq(n)
    compte = 0
    for _ in 1:n
        x = rand(); y = rand()
        if x*x + y*y <= 1.0
            compte += 1
        end
    end
    return 4 * compte / n
end

# ╔═╡ d5134c8a-929c-44c1-a17e-a0f86546cb2a
md"""
**Q1.** Écrivez `pi_faux(n)` en mettant naïvement `@threads` sur la boucle avec un `compte` partagé. Lancez-le plusieurs fois. Constatez-vous le même symptôme qu'à la section 4 (faux + non déterministe) ?
"""

# ╔═╡ 7e2b49d5-0453-46a3-a2ea-eb8b9d04963a
# Q1 : à compléter — la version BUGGÉE, pour constater le problème
function pi_faux(n)
    compte = 0
    @threads for _ in 1:n
        x = rand(); y = rand()
        if x*x + y*y <= 1.0
            compte += 1      # ⚠️ partagé
        end
    end
    return 4 * compte / n
end

# ╔═╡ 86c65859-2a47-4ebf-8c74-ae19cfaf9c44
let
    for _ in 1:5
        println(pi_faux(10_000_000))
    end
end

# ╔═╡ cd8c98c0-b68a-4e57-bfd5-09194e2a16a7
md"""
**Q2.** Corrigez avec la méthode des **accumulateurs locaux** (un compteur par thread, réduction à la fin). Vérifiez que le résultat est stable et proche de π.
"""

# ╔═╡ c9eb0cfe-1a82-4e52-b1e6-dc185dbdf953
# Q2 : à compléter — la version CORRECTE
function pi_par(n)
    comptes = zeros(Int, Threads.nthreads())
    @threads for _ in 1:n
        x = rand(); y = rand()
        if x*x + y*y <= 1.0
            comptes[Threads.threadid()] += 1
        end
    end
    return 4 * sum(comptes) / n
end

# ╔═╡ 3ab8b733-237c-45a2-a98c-ff070c5bc74f
let
    println("π (référence) : ", π)
    for _ in 1:4
        println("pi_par : ", pi_par(10_000_000))
    end
end

# ╔═╡ 1c004e8f-d248-478b-829d-079a3e8b7fa2
md"""
**Q3.** Mesurez `pi_seq` puis `pi_par` avec `@btime`. Quel facteur d'accélération obtenez-vous ? Est-il égal au nombre de threads ? Sinon, **pourquoi** (Amdahl : qu'est-ce qui reste séquentiel ici) ?
"""

# ╔═╡ a31b7b49-5c0f-4252-9885-b9c3316eeab6
# Q3 : votre mesure
let
    print("séquentiel : "); @btime pi_seq(10_000_000)
    print("parallèle  : "); @btime pi_par(10_000_000)
    println("threads : ", Threads.nthreads())
end

# ╔═╡ 63634297-55c3-432c-9e5d-9412472d8359
md"""
**Q4 (ouverture, pour le module GPU).** Le Monte-Carlo est *embarrassingly parallel* : aucun tirage ne dépend d'un autre. C'est le cas **idéal** pour un GPU, qui peut lancer des **milliers** de threads. Mais il y a un piège qu'on n'a pas traité : **le générateur aléatoire `rand()` est-il sûr quand des milliers de threads l'appellent en même temps ?** Gardez la question — c'est un vrai sujet sur GPU.
"""

# ╔═╡ 8eaf6ae9-aebb-4ef2-b13d-62cae72c4a78
md"""
## 6. Bilan

| Idée | Ce qu'on a vu |
|---|---|
| **GIL Python** | `threading` ne parallélise pas le CPU ; Julia n'a pas de GIL → vrai parallélisme mémoire partagée |
| **`@threads` sans partage** | écrire dans des cases différentes : sûr et efficace |
| **Race condition** | `s += ...` partagé → résultat faux *et* non déterministe, **sans erreur** |
| **Réparer** | accumulateurs locaux + réduction (✅✅) ; atomiques (✅ mais lent) |
| **Amdahl** | l'accélération ≠ nombre de threads ; la part séquentielle plafonne |

### À retenir

> Le parallélisme en mémoire partagée est puissant *et* dangereux : le compilateur ne vous protège pas des *race conditions*. La discipline — **minimiser le partage** — n'est pas une optimisation, c'est une condition de correction. Et c'est exactement la discipline qu'exigera le GPU.

### La suite

La **hiérarchie mémoire** (L1/L2/L3/RAM), pourquoi le CPU passe son temps à *attendre la mémoire*, le *false sharing* entrevu aujourd'hui, et le message central du ML : **batcher transforme un problème limité par la mémoire en un problème limité par le calcul** — sur CPU, puis à fond sur **GPU**, où l'on portera enfin le Monte-Carlo.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─79d9e4bb-4e4b-4d59-83a2-7b03d73c6b0d
# ╟─43328603-10db-498a-bd37-3f7167c76c97
# ╠═36881278-4c1e-430d-8e68-bf043782b8a3
# ╠═6186c4ef-f768-4556-ba3c-43ff5975d5ea
# ╟─367ad1ec-8ac8-4e9b-99d9-9aafd2fa7fa9
# ╟─25b0c9b5-0a5a-450a-b686-8b5cb2bb39e4
# ╟─27632886-8768-498a-8544-8a12dd300b6c
# ╠═4c5addbf-2a8f-455d-9364-2a8cbc60e8c6
# ╠═af9c646c-9738-4ee0-8afa-4275f91666f9
# ╠═a6417f05-1d4a-41c0-96bb-2dd9d7f66178
# ╠═00f33fed-d611-4d0a-b6e8-b6e121331667
# ╠═46f8d94c-59de-4334-9ce7-3df3c9ef027e
# ╟─37a835de-b514-429a-a71c-fee0ed868de5
# ╟─e5eca89b-c38c-46c2-8355-801b6d1a1987
# ╠═c683816e-ae0b-4c41-b730-14a84bac69df
# ╠═5268900e-a75e-4651-9692-1fcefe334e09
# ╟─2bb190de-2f19-41a9-bb35-0122f4440cb1
# ╠═3662368f-7669-439b-a879-dfa8bd39e327
# ╟─c3a7201f-dec2-4031-be82-b71e3e30bf0b
# ╟─c5c4b1ab-fd6f-407c-b4cd-df77249d85f2
# ╠═02945be2-9202-4973-b755-ae827fc03dff
# ╠═7044c222-4a99-484d-ad76-29bc5ed056d7
# ╟─f9287ed8-08f8-4681-87d9-6e25c117e9b6
# ╟─18174e65-4a39-48cc-ab55-289da50e0f67
# ╠═c0d3dfab-8f26-4359-9196-a99c76f1553d
# ╠═e027d2e9-230e-418b-81b1-8c78e3314048
# ╟─a990541e-c4f7-4e90-971c-7dbb0bb224c8
# ╠═98c2ad45-47d3-41f5-b077-1dc0b8d3f3bf
# ╟─191e9b94-e2a1-49dd-a689-30754d41a247
# ╟─c77e2fc7-56e1-4c64-9776-899fd97d661e
# ╠═6ca5e3d9-ebf5-45ff-9886-ab465e3e51e1
# ╟─d5134c8a-929c-44c1-a17e-a0f86546cb2a
# ╠═7e2b49d5-0453-46a3-a2ea-eb8b9d04963a
# ╠═86c65859-2a47-4ebf-8c74-ae19cfaf9c44
# ╟─cd8c98c0-b68a-4e57-bfd5-09194e2a16a7
# ╠═c9eb0cfe-1a82-4e52-b1e6-dc185dbdf953
# ╠═3ab8b733-237c-45a2-a98c-ff070c5bc74f
# ╟─1c004e8f-d248-478b-829d-079a3e8b7fa2
# ╠═a31b7b49-5c0f-4252-9885-b9c3316eeab6
# ╟─63634297-55c3-432c-9e5d-9412472d8359
# ╟─8eaf6ae9-aebb-4ef2-b13d-62cae72c4a78
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
