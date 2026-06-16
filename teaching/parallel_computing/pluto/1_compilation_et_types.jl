### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 86593a2f-41c2-464d-905a-70b368785d81
begin
    using BenchmarkTools   # @btime, @benchmark : mesures fiables
    using PlutoUI          # sliders, mise en forme
    using Printf
    md"Paquets chargés ✓"
end

# ╔═╡ ab181e11-6516-4169-9189-fb4488f4c8d3
md"""
# Calcul parallèle — Interprété vs compilé
## Pourquoi le même calcul peut être 100× plus rapide

**ENSAI 3A — Julia comme banc d'essai, Python comme point de comparaison**

---

Le fil de tout le cours tient en une question :

> *Pourquoi le même calcul, mathématiquement identique, peut-il être des dizaines voire des centaines de fois plus rapide selon la façon dont on l'écrit ?*

Aujourd'hui on répond avec trois ingrédients :

1. **Interprété vs compilé** — ce qui se passe vraiment entre votre code et le processeur.
2. **La spécialisation par les types** — pourquoi Julia peut être aussi rapide que du C.
3. **Mesurer proprement** — un chiffre de performance ne vaut que si le protocole de mesure est correct.

Le parallélisme (threads, GPU) viendra dans les modules suivants. Mais on ne peut pas paralléliser intelligemment un code dont on ne comprend pas déjà le coût séquentiel.
"""

# ╔═╡ e4c6629b-29ce-4e63-b529-04bf1020ee4f
md"""
## 0. Mise en route

Cette cellule installe et charge les paquets dont on a besoin. La **première exécution** peut prendre une à deux minutes (téléchargement + précompilation) : c'est déjà une illustration du cours — Julia compile.
"""

# ╔═╡ 9c279c3b-f2c3-4a98-8f21-b21190182e5b
md"""
## 1. Des écarts de plusieurs ordres de grandeur

Commençons par la démonstration la plus simple possible : **sommer un grand vecteur de nombres**.

On va écrire ce calcul de trois façons :

- une **boucle Python pure** (qu'on simulera mentalement / comparera) ;
- la version **`numpy`** ;
- une **boucle Julia**.

Le calcul est *exactement le même*. Les temps, eux, ne le sont pas du tout.
"""

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81176
# Un vecteur de 10 millions de Float64
const N = 10_000_000

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81177
const v = rand(N)

# ╔═╡ 5ea349ac-02a9-420c-a600-8ebfd7a81178
summary(v)

# ╔═╡ 26289681-0477-4670-950f-8351be0fdd5a
# Somme par une boucle Julia "à la main"
function somme_boucle(x)
    s = zero(eltype(x))
    for i in eachindex(x)
        s += x[i]
    end
    return s
end

# ╔═╡ da96898d-efc5-4e47-a73e-8e0a991bc0f8
somme_boucle(v)   # vérifions que ça marche

# ╔═╡ bb0ace77-07a4-4504-80c1-edc793f3f899
md"""
### Le point de comparaison Python

En Python pur, le même calcul s'écrirait :

```python
def somme_boucle(x):
    s = 0.0
    for xi in x:
        s += xi
    return s
```

Sur 10 millions d'éléments, cette boucle Python prend typiquement **~1 seconde**. La boucle Julia ci-dessus prend quelques **millisecondes** — soit un facteur de l'ordre de **100×**, pour un code qui *se ressemble ligne pour ligne*.

`numpy.sum(x)`, lui, est rapide (~10 ms) — mais parce qu'il **ne s'exécute pas en Python** : il appelle une routine C compilée. On y revient en section 4.

La question devient donc : **pourquoi la boucle Julia est-elle rapide alors que la boucle Python est lente ?**
"""

# ╔═╡ 22d3648b-a71a-4299-8e1b-46f032a74046
md"""
## 2. Interprété vs compilé : observer la compilation

Julia permet d'**inspecter** chaque étape entre le code source et l'assembleur.

Prenons une fonction délibérément triviale pour que la sortie reste lisible.
"""

# ╔═╡ a51ea6fa-9f15-453f-bcea-852c7e71a354
f(x) = 2x + 1

# ╔═╡ e8d202f2-8080-4e46-aa29-157c5a707c13
md"""
### Étape 1 — le code « abaissé » (*lowered*)

Julia réécrit d'abord votre code sous une forme canonique en instructions élémentaires (SSA). C'est encore indépendant des types.
"""

# ╔═╡ 26118708-e6fa-4cdb-a5e9-7a72a376b84f
@code_lowered f(3.0)

# ╔═╡ c3c88228-8ec2-4a08-9384-7da546be08fb
md"""
### Étape 2 — le code typé

C'est l'étape **décisive**. Julia connaît maintenant le type de l'argument (`Float64`) et **propage les types** dans toute la fonction : chaque opération sait sur quoi elle porte.

**C'est ici que se joue « une compilation par type ».** La *même* fonction `f` produit un code différent selon le type des arguments : `f(3.0)` (`Float64`) et `f(3)` (`Int`) déclenchent **deux versions compilées distinctes**, chacune spécialisée pour son type. (Julia appelle ça une *spécialisation* / *method instance* : il en compile une nouvelle à la **première** rencontre de chaque combinaison de types, puis la réutilise.)

Comparez les deux sorties :
"""

# ╔═╡ c14f0bba-a2a2-457d-a400-35224ec23b34
@code_typed f(3.0)

# ╔═╡ eb5c0a66-0e2e-4eba-9cc6-0b6f6acd4fb6
@code_typed f(3)

# ╔═╡ 3e4c87e0-e1aa-4593-8f71-a57b44ef14fe
md"""
### Étape 3 — l'assembleur

On descend jusqu'aux instructions du processeur. La compilation par type y devient **directement observable** :

- `f(3.0)` (`Float64`) → instructions **flottantes** (`vmulsd`, `vaddsd`…) ;
- `f(3)` (`Int`) → instructions **entières** (`lea`, `add`…).

Deux types → **deux codes machine différents**, quelques lignes chacun, sans boucle ni « interpréteur ». La preuve, au niveau le plus bas, qu'il y a bien une version compilée **par type**.
"""

# ╔═╡ cf7f8128-23b8-4c2e-a122-d2705f2c43c6
@code_native debuginfo=:none f(3.0)

# ╔═╡ d5000000-0000-4d00-9b00-00000000d001
@code_native debuginfo=:none f(3)

# ╔═╡ cb125025-5a35-4e36-bf13-7efc3e29abd4
md"""
### Ce qu'il faut retenir

- **Python (CPython)** garde votre code sous forme de *bytecode* et le fait exécuter par une grosse boucle d'interprétation. Chaque `+` passe par : « quel est le type de l'objet ? a-t-il une méthode `__add__` ? … ». Ce **coût par opération** est ce qui pénalise la boucle.
- **Julia** compile, *à la première utilisation et pour chaque combinaison de types*, une version spécialisée native. Une fois compilée, `2x+1` sur des `Float64` n'est littéralement que deux instructions machine.

C'est le sens de « Julia est compilé » : pas un compilateur lancé à la main avant l'exécution, mais une compilation **juste-à-temps (JIT)**, déclenchée par les types des arguments.

> **Distinction clé — quand le type est-il connu ?**
> - **À l'exécution** (Python) : le type est redécouvert à *chaque* opération → un surcoût payé à chaque itération → **lent**.
> - **À la compilation** (Julia/JIT) : le type est connu une fois, le code natif est spécialisé pour lui, le test de type **disparaît** du code chaud → **rapide**.
>
> Cette opposition **temps de compilation vs temps d'exécution** réapparaîtra pour le **dispatch**, les **threads** et le **GPU**.
"""

# ╔═╡ fad9827d-f3db-43ae-b0bb-b58aca083e79
md"""
## 3. Le piège n°1 : l'instabilité de type

« Julia est rapide » est faux si le compilateur **ne peut pas** déterminer les types. Le cas le plus courant : une **variable globale non typée**.

Voici la même somme, mais qui lit une variable globale.
"""

# ╔═╡ fa9596a7-23bd-4675-998b-af232a7b08e8
# variable globale (type non fixé du point de vue du compilateur)
glob = rand(1000)

# ╔═╡ fa9596a7-23bd-4675-998b-af232a7b08e9
function somme_globale()
    s = 0.0
    for i in eachindex(glob)   # glob est une globale
        s += glob[i]
    end
    return s
end

# ╔═╡ 41165e5b-f988-4698-97d1-04ca43eb7964
function somme_argument(x)
    s = 0.0
    for i in eachindex(x)
        s += x[i]
    end
    return s
end

# ╔═╡ 761ce2e0-503b-48b1-9a5d-2b2cce7f4f45
md"""
L'outil clé est `@code_warntype` : il colore en **rouge** les endroits où Julia ne connaît pas le type (`Any`). Le rouge = du code qui retombe dans un comportement dynamique, comparable à Python.

Regardez la version globale (du rouge / `Any`) :
"""

# ╔═╡ ba6f002b-32d3-4327-a14e-ad049956e1ec
@code_warntype somme_globale()

# ╔═╡ 8d4f26b4-4d42-4291-8c31-0891ba765a37
md"""
…puis la version qui prend son tableau **en argument** (tout est typé, pas de rouge) :
"""

# ╔═╡ 0fcbe125-0551-42b2-93b7-2fe0f6e65bb2
@code_warntype somme_argument(glob)

# ╔═╡ bd7de916-7afd-4b63-81e3-2f82fc00b935
md"""
### La leçon, mesurée

Les deux font le même calcul. Mesurons l'écart.
"""

# ╔═╡ f53445b9-d198-4ad4-9414-140afdb6460c
@benchmark somme_globale()

# ╔═╡ 20c2f0e8-b528-40cf-94b5-e308ae298137
@benchmark somme_argument($glob)

# ╔═╡ 46d41ce1-700e-41eb-8ae4-02a37802725b
md"""
**Règle d'or Julia** : mettez le travail dans des **fonctions** qui reçoivent leurs données **en arguments**. C'est ce qui permet la spécialisation par les types — donc la vitesse. Une boucle « au niveau global » dans un script Julia est une erreur fréquente lorsqu'on arrive de Python.
"""

# ╔═╡ 260d7cdf-31b5-47d5-8ea5-ccc76e270dbe
md"""
## 4. Pourquoi numpy est rapide (et ce que ça cache)

Reprenons Python. `numpy.sum(x)` est ~100× plus rapide qu'une boucle Python. Pourquoi ?

Parce que `numpy` **ne fait pas la boucle en Python**. Il stocke les données dans un tableau contigu typé, et délègue la somme à une **fonction C compilée** (les fichiers `.so` / `.pyd` de numpy, eux-mêmes issus de C). Python ne sert qu'à *appeler* ce C.

Conséquence importante :

- Tant que vous restez dans des opérations numpy « vectorisées » (`x + y`, `x.sum()`, `x @ y`), vous êtes rapide.
- **Dès que vous écrivez une boucle Python autour d'éléments numpy**, vous repayez le coût de l'interpréteur à chaque tour, et tout s'effondre.

Julia n'a pas cette frontière : la boucle *et* le « vectorisé » sont le même langage compilé. C'est ce qu'on appelle le *two-language problem* — Python le résout en écrivant les parties rapides dans un autre langage (C, Cython, Numba) ; Julia tente de l'éliminer.

### Et Numba ?

`@numba.njit` ajoute un JIT à Python : il compile certaines fonctions numériques en code natif, et peut rendre une boucle Python presque aussi rapide que Julia. Mais c'est un **sous-ensemble** de Python (types limités, pas tout objet Python n'est supporté), et il faut le déclencher explicitement. C'est exactement la mécanique de Julia… rajoutée par-dessus Python pour un domaine restreint. Bon contrepoint : « JIT » n'est pas magique, c'est *spécialiser par les types puis compiler*.
"""

# ╔═╡ d02cd083-dea1-417f-a941-8eac616197a6
md"""
## 5. Mesurer proprement — l'art du benchmark

Tout ce qui précède repose sur des chiffres. Un chiffre faux est pire que pas de chiffre. Voici les pièges, et comment Julia les gère.
"""

# ╔═╡ 99808d58-8875-42ed-bd94-c7e06ab36c10
md"""
### Piège A — mesurer la compilation au lieu du calcul

`@time` inclut, au **premier appel**, le temps de **compilation JIT**. Exécutez la cellule suivante : le premier `@time` est gonflé (allocations énormes, temps « compilation »), le second est le vrai temps.
"""

# ╔═╡ a8128f4d-9cdc-4cd2-8025-33c5935757e0
let
    g(x) = sum(abs2, x)        # nouvelle fonction, jamais compilée
    @time g(v)                 # 1er appel : inclut la compilation
    @time g(v)                 # 2e appel : le vrai coût
end

# ╔═╡ 46c146f1-d1e4-4797-87bc-2c03a0185daf
md"""
### Piège B — ne mesurer qu'une seule fois

Une mesure unique est polluée par le bruit (OS, fréquence CPU, cache). Il faut **répéter** et regarder la distribution. C'est tout l'intérêt de `BenchmarkTools` :

- `@btime` : affiche le **minimum** (le plus représentatif du coût « pur »), plus les allocations.
- `@benchmark` : la distribution complète (min / médiane / moyenne / max).

⚠️ **Toujours interpoler les variables avec `\$`** (`@btime f(\$x)`) pour que la variable ne soit pas traitée comme une globale — sinon vous mesurez le piège de la section 3, pas votre fonction.
"""

# ╔═╡ ca49af93-0baa-47ee-a4ec-3849a1cacffb
@benchmark somme_boucle($v)

# ╔═╡ e44cfddd-c301-4816-924c-24b4a51e81b0
md"""
### Piège C — comparer des secondes plutôt qu'un débit

« 4 ms » n'a pas de sens dans l'absolu. Rapporté à la taille du problème, cela devient un **débit** comparable d'une machine à l'autre : ici, des **Go/s** de mémoire lue. C'est le bon langage pour le module sur la mémoire (où elle devient le facteur limitant).
"""

# ╔═╡ b973c77a-e95b-4324-a475-b57088b46aa6
let
    t = @belapsed somme_boucle($v)        # secondes
    octets = sizeof(v)                     # octets lus
    gbs = octets / t / 1e9
    @sprintf("Temps : %.3f ms  |  Débit : %.1f Go/s", t*1e3, gbs)
end

# ╔═╡ 49ea0717-5e4a-44df-a895-11b9d55c747f
md"""
## 6. Activité guidée — Monte-Carlo de π

On met tout en pratique sur un classique : estimer π en tirant des points au hasard dans le carré $[0,1]^2$ et en comptant la fraction qui tombe dans le quart de disque.

$$\pi \approx 4 \times \frac{\#\{x^2+y^2 \le 1\}}{n}$$

Ce sera l'**exemple conducteur** du cours : d'abord en séquentiel, puis en multi-thread (CPU), puis sur GPU.
"""

# ╔═╡ 760cd379-4178-4e7f-9695-a1d709b36969
function estime_pi(n)
    compte = 0
    for _ in 1:n
        x = rand(); y = rand()
        if x*x + y*y <= 1.0
            compte += 1
        end
    end
    return 4 * compte / n
end

# ╔═╡ 8adcfe41-1d0d-475a-968c-5a5a971a142b
estime_pi(10_000_000)

# ╔═╡ 855d77ed-ca4b-407a-b225-2be1a2c499f5
md"""
**Exercices.** Répondez dans les cellules suivantes (modifiez le code, Pluto réévalue automatiquement).

**Q1.** Mesurez `estime_pi` avec `@btime` pour `n = 10⁷`. Combien de temps ? Combien d'allocations ? (Indice : `rand()` sans argument n'alloue pas — c'est voulu.)
"""

# ╔═╡ 3cb6763c-dc00-4c09-94b5-289f773dc460
# Q1 : votre mesure ici
@btime estime_pi(10_000_000)

# ╔═╡ f15bd89d-47f6-4c87-a3d9-17078b722f1e
md"""
**Q2.** Vérifiez la stabilité de type avec `@code_warntype estime_pi(1000)`. Voyez-vous du rouge ? Pourquoi le résultat est-il `Int` ou `Float64` selon les branches ?
"""

# ╔═╡ 654b39b4-b592-4066-b519-0783a704f637
# Q2 : votre analyse ici
@code_warntype estime_pi(1000)

# ╔═╡ 49546ce1-11ee-4cc8-b412-24b9e933a728
md"""
**Q3 (au tableau / discussion).** L'erreur sur π décroît comme $1/\sqrt{n}$. Pour gagner un chiffre décimal il faut donc ~100× plus de points. **Pourquoi est-ce le candidat *parfait* pour le parallélisme ?** (Indice : chaque tirage est totalement indépendant des autres — on dit *embarrassingly parallel*. Gardez cette idée pour le module sur les threads.)
"""

# ╔═╡ b7792bef-9262-42b4-98d0-407113ded36c
# Q3 : espace libre pour expérimenter (faire varier n et observer l'erreur)
let
    for n in (10^4, 10^5, 10^6, 10^7, 10^8)
        err = abs(estime_pi(n) - π)
        println(@sprintf("n = 10^%d   erreur = %.5f", round(Int, log10(n)), err))
    end
end

# ╔═╡ 765124ae-9c5d-436c-ac54-d3cb26587b87
md"""
## 7. Bilan

| Idée | Ce qu'on a vu |
|---|---|
| **Interprété (Python)** | bytecode + boucle d'interprétation → coût par opération |
| **Compilé JIT (Julia)** | spécialisation par types → code natif, pas de surcoût par opération |
| **numpy / Numba** | rapides car ils *sortent* de l'interpréteur Python (C compilé / JIT restreint) |
| **Stabilité de type** | la condition pour que Julia soit rapide ; `@code_warntype`, pas de globales |
| **Benchmark** | `@btime`/`@benchmark`, interpoler les variables, raisonner en débit |

### À retenir

> La vitesse ne vient pas du langage « en soi », mais de **ce que la machine doit faire par opération**. Compiler et spécialiser supprime le surcoût ; il ne restera plus, une fois la mémoire abordée, qu'un seul ennemi : **le temps d'aller chercher les données en mémoire**.

### La suite

D'abord le **multiple dispatch** de Julia (vs l'orienté objet de Python) ; puis les **threads** et un cas classique : **la somme parallèle qui donne un résultat faux**.
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

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "db49b271b4149145d21642645bfe62ddc84408ba"

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
# ╟─9c279c3b-f2c3-4a98-8f21-b21190182e5b
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81176
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81177
# ╠═5ea349ac-02a9-420c-a600-8ebfd7a81178
# ╠═26289681-0477-4670-950f-8351be0fdd5a
# ╠═da96898d-efc5-4e47-a73e-8e0a991bc0f8
# ╟─bb0ace77-07a4-4504-80c1-edc793f3f899
# ╟─22d3648b-a71a-4299-8e1b-46f032a74046
# ╠═a51ea6fa-9f15-453f-bcea-852c7e71a354
# ╟─e8d202f2-8080-4e46-aa29-157c5a707c13
# ╠═26118708-e6fa-4cdb-a5e9-7a72a376b84f
# ╟─c3c88228-8ec2-4a08-9384-7da546be08fb
# ╠═c14f0bba-a2a2-457d-a400-35224ec23b34
# ╠═eb5c0a66-0e2e-4eba-9cc6-0b6f6acd4fb6
# ╟─3e4c87e0-e1aa-4593-8f71-a57b44ef14fe
# ╠═cf7f8128-23b8-4c2e-a122-d2705f2c43c6
# ╠═d5000000-0000-4d00-9b00-00000000d001
# ╟─cb125025-5a35-4e36-bf13-7efc3e29abd4
# ╟─fad9827d-f3db-43ae-b0bb-b58aca083e79
# ╠═fa9596a7-23bd-4675-998b-af232a7b08e8
# ╠═fa9596a7-23bd-4675-998b-af232a7b08e9
# ╠═41165e5b-f988-4698-97d1-04ca43eb7964
# ╟─761ce2e0-503b-48b1-9a5d-2b2cce7f4f45
# ╠═ba6f002b-32d3-4327-a14e-ad049956e1ec
# ╟─8d4f26b4-4d42-4291-8c31-0891ba765a37
# ╠═0fcbe125-0551-42b2-93b7-2fe0f6e65bb2
# ╟─bd7de916-7afd-4b63-81e3-2f82fc00b935
# ╠═f53445b9-d198-4ad4-9414-140afdb6460c
# ╠═20c2f0e8-b528-40cf-94b5-e308ae298137
# ╟─46d41ce1-700e-41eb-8ae4-02a37802725b
# ╟─260d7cdf-31b5-47d5-8ea5-ccc76e270dbe
# ╟─d02cd083-dea1-417f-a941-8eac616197a6
# ╟─99808d58-8875-42ed-bd94-c7e06ab36c10
# ╠═a8128f4d-9cdc-4cd2-8025-33c5935757e0
# ╟─46c146f1-d1e4-4797-87bc-2c03a0185daf
# ╠═ca49af93-0baa-47ee-a4ec-3849a1cacffb
# ╟─e44cfddd-c301-4816-924c-24b4a51e81b0
# ╠═b973c77a-e95b-4324-a475-b57088b46aa6
# ╟─49ea0717-5e4a-44df-a895-11b9d55c747f
# ╠═760cd379-4178-4e7f-9695-a1d709b36969
# ╠═8adcfe41-1d0d-475a-968c-5a5a971a142b
# ╟─855d77ed-ca4b-407a-b225-2be1a2c499f5
# ╠═3cb6763c-dc00-4c09-94b5-289f773dc460
# ╟─f15bd89d-47f6-4c87-a3d9-17078b722f1e
# ╠═654b39b4-b592-4066-b519-0783a704f637
# ╟─49546ce1-11ee-4cc8-b412-24b9e933a728
# ╠═b7792bef-9262-42b4-98d0-407113ded36c
# ╟─765124ae-9c5d-436c-ac54-d3cb26587b87
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
