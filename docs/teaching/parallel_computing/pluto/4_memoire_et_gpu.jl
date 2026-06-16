### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ aa07d25d-4f02-4013-95a0-49f72ddf7323
md"""
# Calcul parallèle — Mémoire et GPU
## La mémoire, le batching, et le GPU

**ENSAI 3A**

---

Jusqu'ici : *compiler et spécialiser* supprime le surcoût **par opération** ; *paralléliser* exige de **minimiser le partage** (sinon : résultats faux).

Il reste un dernier ennemi, le plus important en pratique :

> **Le processeur passe le plus clair de son temps à *attendre* la mémoire.**

Aujourd'hui :

1. La **hiérarchie mémoire** (registres → L1 → L2 → L3 → RAM) et ses ordres de grandeur.
2. La **localité** : pourquoi le *même* calcul est 2 à 10× plus lent selon l'ordre des accès (et le piège Julia *column-major* vs numpy *row-major*).
3. Le message du ML : **batcher transforme un problème limité par la mémoire en un problème limité par le calcul**.
4. Le **GPU**, où ce principe devient vital — avec le Monte-Carlo de π porté sur carte graphique.
"""

# ╔═╡ ee0a05ee-935e-42c3-b567-5624fb902747
md"""
## 0. Mise en route
"""

# ╔═╡ 3b62bc4c-9765-4424-b7ee-2800c085103a
begin
    using BenchmarkTools
    using PlutoUI
    using Printf
    using LinearAlgebra
    using CUDA          # si pas de GPU, le paquet se charge quand même ;
                        # c'est CUDA.functional() qui dira s'il est utilisable
    md"Paquets chargés ✓"
end

# ╔═╡ e60ad78b-4669-47b1-b819-443e4454f585
md"""
## 1. La hiérarchie mémoire — les ordres de grandeur en jeu

Un cœur de processeur peut faire une addition flottante en **moins d'une nanoseconde**. Mais aller chercher une donnée en RAM prend **~100 nanosecondes**. Pendant ce temps, le cœur pourrait avoir fait **des centaines** d'opérations. S'il attend, il ne fait rien.

Pour atténuer ça, le matériel intercale des **caches** de plus en plus petits et rapides entre le cœur et la RAM :

| Niveau | Taille typique | Latence approx. | Analogie (si L1 = 1 s) |
|---|---|---|---|
| Registres | ~quelques centaines d'octets | ~0 | instantané |
| Cache **L1** | ~32–64 Ko | ~1 ns | **1 seconde** |
| Cache **L2** | ~256 Ko–1 Mo | ~4 ns | 4 secondes |
| Cache **L3** | ~8–32 Mo | ~15 ns | 15 secondes |
| **RAM** | ~16–64 Go | ~100 ns | **~2 minutes** |
| Disque SSD | To | ~100 µs | **~1 jour** |

L'idée à retenir : **plus c'est gros, plus c'est loin, plus c'est lent.** Le cache fonctionne sur un pari — la **localité** :

- **localité temporelle** : si on vient d'utiliser une donnée, on va probablement la réutiliser bientôt → on la garde près.
- **localité spatiale** : si on utilise une donnée, on va probablement utiliser **ses voisines** → le cache charge la mémoire par **lignes** (~64 octets, soit 8 `Float64`) d'un coup.

Tout l'art du code performant tient à **respecter ce pari**.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c001
md"""
### Voir les paliers : l'expérience du « cache cliff »

Cette table, on peut la **mesurer**. On somme un tableau de taille croissante, en le reparcourant plusieurs fois (pour qu'il reste « chaud » dans le plus petit cache qui le contient), et on regarde le **temps par accès** :

- tant que le tableau **tient dans L1**, chaque accès est quasi gratuit ;
- dès qu'il **déborde** vers L2, puis L3, puis la RAM, le temps par accès grimpe **par paliers** — on *voit* la hiérarchie apparaître.

⚠️ Les seuils dépendent de **votre** machine : repérez les **sauts**, pas les valeurs absolues.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c002
# Somme `a` un grand nombre de fois : l'empreinte mémoire reste celle de `a`,
# mais on multiplie les accès → on mesure la vitesse du niveau de cache qui contient `a`.
function somme_repetee(a, passes)
    s = zero(eltype(a))
    for _ in 1:passes
        @inbounds @simd for i in eachindex(a)
            s += a[i]
        end
    end
    return s
end

# ╔═╡ cb000000-0000-4c00-9a00-00000000c003
let
    println("empreinte | temps par accès (ns)")
    println("-"^32)
    for ko in (8, 24, 64, 256, 1024, 4096, 16384, 65536)
        n = ko * 128                       # Ko → nombre de Float64 (1024/8 = 128)
        a = rand(n)
        passes = max(1, 100_000_000 ÷ n)   # ≈ volume total d'accès constant
        t = @belapsed somme_repetee($a, $passes)
        ns = t / (passes * n) * 1e9
        empreinte = ko < 1024 ? @sprintf("%4d Ko", ko) : @sprintf("%4d Mo", ko ÷ 1024)
        println(@sprintf("%9s | %14.3f", empreinte, ns))
    end
end

# ╔═╡ 58cf2a8a-433f-4737-8116-044f7f8eada1
md"""
## 2. La localité, mesurée : parcourir une matrice dans le bon sens

Voici **le** TP central de ce module. On somme tous les éléments d'une matrice de deux façons :

- en parcourant **colonne par colonne** ;
- en parcourant **ligne par ligne**.

C'est **exactement le même calcul**, le même nombre d'additions. Pourtant les temps diffèrent largement. Pourquoi ? À cause de la façon dont la matrice est **rangée en mémoire**.

> **Point crucial — Julia range les matrices par *colonnes* (column-major).** Deux éléments d'une même *colonne* sont voisins en mémoire. C'est **l'inverse de C / numpy**, qui rangent par *lignes* (row-major). Si vous transposez mentalement vos réflexes numpy, vous serez systématiquement à contre-sens en Julia.
"""

# ╔═╡ 3218651a-5b26-42d1-b330-917d0801ed44
begin
    const A = rand(8_000, 8_000)   # 64 M de Float64 ≈ 512 Mo : déborde largement le cache
    size(A), Base.summarysize(A) ÷ 1_000_000, "Mo"
end

# ╔═╡ 2126f361-6be9-4f14-a358-6a034213fa4f
# Parcours COLONNE par colonne : l'indice de ligne (i) varie le plus vite.
# En column-major, A[i,j] et A[i+1,j] sont VOISINS en mémoire → localité spatiale respectée.
function somme_colonnes(M)
    s = zero(eltype(M))
    @inbounds for j in axes(M, 2)        # pour chaque colonne
        for i in axes(M, 1)              # on descend la colonne
            s += M[i, j]
        end
    end
    return s
end

# ╔═╡ 6908e575-a27a-43b8-b7ee-2cf0bce7f8fc
# Parcours LIGNE par ligne : l'indice de colonne (j) varie le plus vite.
# A[i,j] et A[i,j+1] sont distants de "une colonne entière" en mémoire → on saute partout.
function somme_lignes(M)
    s = zero(eltype(M))
    @inbounds for i in axes(M, 1)        # pour chaque ligne
        for j in axes(M, 2)             # on traverse la ligne
            s += M[i, j]
        end
    end
    return s
end

# ╔═╡ 43f403f1-a036-4454-8670-327d079dd5cc
somme_colonnes(A) ≈ somme_lignes(A)      # même résultat mathématique

# ╔═╡ cdad600b-7259-4388-a83b-0a2f42f83aa1
let
    print("colonnes (bon sens) : "); @btime somme_colonnes($A)
    print("lignes  (à contre-sens) : "); @btime somme_lignes($A)
end

# ╔═╡ 62969c4a-d889-4322-8923-725b34f839dc
md"""
### Ce qu'on vient de voir

Le **même calcul**, juste réordonné, est plusieurs fois plus lent.

- **`somme_colonnes`** : on lit la mémoire dans l'ordre où elle est rangée. Chaque ligne de cache chargée (64 octets = 8 `Float64`) est **entièrement utilisée** avant de passer à la suivante. Le pari de la localité spatiale est gagné.
- **`somme_lignes`** : à chaque pas, on saute de toute la longueur d'une colonne. On charge une ligne de cache pour n'en utiliser **qu'un seul élément**, puis on la jette. On repaye le coût d'accès mémoire bien plus souvent.

> **Règle Julia** : la boucle sur le **premier indice** (les lignes) doit être la boucle **intérieure**. C'est l'inverse de C/numpy. Se tromper ne donne pas un résultat faux — juste un code lent, ce qui est plus sournois.
"""

# ╔═╡ 09844b2e-6544-494d-a7c1-50c3d5b9c5d6
md"""
**Activité éclair.** Faites varier la taille ci-dessous avec le curseur. Pour une **petite** matrice qui tient dans le cache, l'écart disparaît (tout est déjà « près »). L'écart n'apparaît que quand les données **débordent du cache**. C'est la preuve directe que c'est bien un effet de cache.
"""

# ╔═╡ 842ce51a-2b62-49e4-a447-1b842c6fe632
@bind taille Slider(100:100:4000, default=2000, show_value=true)

# ╔═╡ 01d52dcd-e327-418c-9906-dddd778b01f0
let
    B = rand(taille, taille)
    tc = @belapsed somme_colonnes($B)
    tl = @belapsed somme_lignes($B)
    octets = Base.summarysize(B) ÷ 1024
    @sprintf("matrice %d×%d (%d Ko) — colonnes : %.2f ms | lignes : %.2f ms | ratio : %.1f×",
             taille, taille, octets, tc*1e3, tl*1e3, tl/tc)
end

# ╔═╡ ad57f7bf-d863-4be8-b5d4-d0dfd0603b69
md"""
## 3. De la localité au batching : le message clé du ML

On arrive au cœur. Introduisons une notion : l'**intensité arithmétique** = nombre d'opérations de calcul effectuées **par octet chargé** depuis la mémoire.

- Intensité **faible** → on passe son temps à charger des données, le calcul attend : on est **limité par la mémoire** (*memory-bound*).
- Intensité **élevée** → on réutilise beaucoup chaque donnée chargée, les unités de calcul tournent à plein : on est **limité par le calcul** (*compute-bound*).

Comparons deux opérations d'algèbre linéaire.
"""

# ╔═╡ 05884f27-4254-4a8e-8177-e38454dc8aac
begin
    const n = 2000
    const Mat = rand(n, n)
    const vec = rand(n)
    const Mat2 = rand(n, n)
    md"Données prêtes."
end

# ╔═╡ dc815f82-3a2e-40a7-b9a1-67daf6c38766
let
    # produit matrice-VECTEUR : chaque coefficient de Mat est lu UNE fois, peu réutilisé.
    # ~2n² opérations pour ~n² données lues → intensité ~constante : MEMORY-BOUND.
    print("matrice × vecteur : "); @btime $Mat * $vec

    # produit matrice-MATRICE : chaque coefficient est réutilisé n fois.
    # ~2n³ opérations pour ~n² données → intensité ∝ n : COMPUTE-BOUND.
    print("matrice × matrice : "); @btime $Mat * $Mat2
end

# ╔═╡ fc30c6c4-6c52-4bad-b28e-5f54f21b4b1c
md"""
### Estimation des ordres de grandeur

Posons les ordres de grandeur pour `n = 2000` :

- **Matrice × vecteur** : ~2·n² ≈ 8 millions d'opérations, pour ~n² ≈ 4 millions de données lues. **Environ 2 opérations par donnée.** Dès qu'on a chargé un coefficient, on s'en sert à peine. Le calcul attend la mémoire.
- **Matrice × matrice** : ~2·n³ ≈ 16 **milliards** d'opérations, pour toujours ~n² données. **Environ n = 2000 opérations par donnée chargée.** Chaque coefficient sert des milliers de fois (c'est le *tiling* / blocage que fait BLAS) → on sature les unités de calcul.

C'est pourquoi le produit matrice-matrice (BLAS niveau 3, `GEMM`) atteint quasiment la **performance crête** de la machine, alors que matrice-vecteur (niveau 2) plafonne à la **bande passante mémoire**.

### Le lien avec le machine learning

Un réseau de neurones, c'est essentiellement des produits matriciels.

- **Inférer sur 1 seul exemple** → des produits matrice-**vecteur** → *memory-bound* → le matériel est sous-utilisé.
- **Inférer sur un *batch* de B exemples** → les B vecteurs s'empilent en une **matrice** → produits matrice-**matrice** → *compute-bound* → le matériel travaille à plein.

> **« On batche pour accélérer »** n'est pas une recette magique : c'est **transformer un problème limité par la mémoire en un problème limité par le calcul**. Le même travail par exemple, mais en réutilisant chaque poids chargé pour tous les exemples du batch au lieu d'un seul.

Et c'est *encore plus* vrai sur GPU — où la pénalité de sous-utilisation est énorme. C'est l'objet de la fin de ce module.
"""

# ╔═╡ ba7c1100-0001-4a01-8b01-aaaaaaaa0001
md"""
### Le voir directement : le débit **par exemple**

⚠️ Le benchmark plus haut comparait *un* matrice-vecteur à *un* matrice-matrice. Mais ils ne font pas la même quantité de travail : le matrice-matrice paraît juste « plus long ». Ça ne **montre** pas l'intérêt du batching.

Le vrai test, c'est : **à exemple donné, combien de temps PAR EXEMPLE ?** On prend une « couche » `W` (n entrées → n sorties) et on l'applique à un *batch* de `B` exemples empilés en colonnes (`W * X`, avec `X` de taille `n×B`). On regarde le temps ramené à **un** exemple quand `B` grandit.
"""

# ╔═╡ ba7c1100-0002-4a01-8b01-aaaaaaaa0002
let
    W = rand(Float32, n, n)          # une couche dense : n → n
    println("batch B | temps total (ms) | temps PAR EXEMPLE (µs) | exemples/s")
    println("-"^66)
    for B in (1, 8, 64, 256, 1024)
        X = rand(Float32, n, B)      # B exemples empilés en colonnes
        t = @belapsed $W * $X        # un seul produit matriciel pour tout le batch
        println(@sprintf("%6d  | %12.2f | %18.2f | %12.0f",
                         B, t*1e3, t/B*1e6, B/t))
    end
end

# ╔═╡ ba7c1100-0003-4a01-8b01-aaaaaaaa0003
md"""
Le temps **total** augmente avec `B` (on calcule plus), mais le temps **par exemple** *chute* : un même poids `W` chargé en mémoire est réutilisé pour les `B` exemples du batch au lieu d'un seul. C'est exactement « passer de matrice-vecteur (memory-bound) à matrice-matrice (compute-bound) » — et c'est *ça*, concrètement, l'accélération du batching.
"""

# ╔═╡ 92cd196c-2cab-4aee-aecb-89f06133486c
md"""
## 4. Le GPU : des milliers de threads, à condition de les alimenter

Un CPU, c'est quelques cœurs très rapides et « intelligents ». Un GPU, c'est **des milliers** de cœurs plus simples, conçus pour faire **la même opération sur des données différentes** en masse (modèle **SIMT** : *Single Instruction, Multiple Threads*).

Deux conséquences :

- **Bande passante mémoire énorme**, et une latence mémoire **cachée par le parallélisme** : quand des milliers de threads attendent la mémoire, le GPU bascule sur d'autres threads prêts. Il faut donc **beaucoup** de travail simultané pour « remplir » la machine (*occupancy*).
- **Un petit problème sous-utilise fortement le GPU.** Pour seulement 100 éléments, le transfert CPU↔GPU et le coût de lancement dominent le calcul lui-même. **Il faut de gros batchs.**

C'est la même conclusion que la section 3, poussée à l'extrême : le GPU n'a d'intérêt que si on lui fournit **massivement** du travail parallèle et régulier.

> ⚙️ **Les cellules suivantes nécessitent un GPU NVIDIA + `CUDA.jl`.** Si vous n'en avez pas, lisez-les : elles sont commentées pour rester compréhensibles, et une version « repli CPU » est fournie plus bas.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c004
md"""
### Caches CPU vs caches GPU : deux stratégies opposées

Le GPU a **lui aussi** une hiérarchie mémoire — le *principe* est identique (la mémoire est loin, on rapproche les données) :

| Niveau | CPU | GPU (NVIDIA) |
|---|---|---|
| le plus proche | registres | registres (par thread) |
| scratchpad / L1 | cache L1 | **shared memory** + L1 (par SM) |
| partagé | caches L2, L3 | cache L2 (entre SM) |
| loin | RAM (DDR) | mémoire globale (VRAM/HBM) |

Mais la **stratégie** est inverse :

- **Comment masquer la latence ?** Le CPU mise sur de **gros caches** (des Mo par cœur) pour *peu* de threads. Le GPU mise sur **un très grand nombre de threads** : quand un *warp* attend la mémoire, le matériel bascule sur un autre warp prêt → il faut **beaucoup** de threads pour masquer l'attente (*occupancy*). C'est précisément ce qu'on entend par « fournir assez de travail au GPU ».
- **Qui gère ?** Les caches CPU sont **automatiques** (transparents). Sur GPU, la *shared memory* est un **scratchpad géré explicitement par le programme** : on décide quelles données y placer (un L1 « manuel »).
- **Localité spatiale.** Côté CPU : lire des cases **voisines** (lignes de cache de 64 o) — c'est le parcours colonne/ligne de la section 2. Côté GPU, l'analogue est la **coalescence** : les 32 threads d'un *warp* doivent lire des adresses **contiguës** pour que le matériel fusionne leurs accès en une seule transaction.

> **Même problème, deux stratégies.** CPU : *gros caches, peu de threads*. GPU : *petits caches, très grand nombre de threads + scratchpad explicite*. La **localité** reste déterminante des deux côtés — on la mesure sur GPU juste après.
"""

# ╔═╡ c3790242-0bd4-4168-858a-3e58d5bb3134
# Le GPU est-il réellement utilisable ? (carte + pilote + CUDA opérationnels)
gpu_dispo = CUDA.functional()

# ╔═╡ e366e8e3-9f59-40bd-bb00-89347808fb1f
if gpu_dispo
    let
        CUDA.versioninfo()
    end
else
    md"⚠️ **Pas de GPU CUDA détecté.** Les cellules GPU afficheront un message ; la section 6 fournit un repli CPU."
end

# ╔═╡ cb000000-0000-4c00-9a00-00000000c005
md"""
### Côté GPU : la coalescence, mesurée

L'analogue GPU du « bon sens de parcours ». On lit le **même** tableau via un tableau d'indices :

- **contigus** (`idx[i] = i`) : des threads voisins lisent des adresses voisines → le matériel **fusionne** leurs lectures en une seule transaction = accès *coalescé* ;
- **à grand pas** (`idx[i] = pas·i`) : des threads voisins lisent **loin** les uns des autres → beaucoup plus de transactions = accès **non coalescé**.

Même nombre de lectures, même résultat — seul l'**ordre des accès** change. C'est exactement la localité spatiale de la section 2, transposée au *warp*.
"""

# ╔═╡ cb000000-0000-4c00-9a00-00000000c006
if gpu_dispo
    let
        N = 20_000_000
        a = CUDA.rand(Float32, N)
        contigu   = CuArray(Int32.(1:N))                              # voisins → coalescé
        pas       = 32
        eparpille = CuArray(Int32.(((0:N-1) .* pas) .% N .+ 1))       # grand pas → non coalescé
        print("accès contigu  (coalescé)     : "); @btime CUDA.@sync $a[$contigu]
        print("accès à grand pas (éparpillé) : "); @btime CUDA.@sync $a[$eparpille]
    end
else
    md"*(démo GPU sautée — pas de carte détectée)*"
end

# ╔═╡ cb000000-0000-4c00-9a00-00000000c007
md"""
L'accès à grand pas est nettement plus lent **pour le même nombre de lectures** : les threads d'un warp ne tombent plus sur la même ligne, le GPU doit émettre bien plus de transactions mémoire. C'est le pendant exact du parcours « à contre-sens » de la section 2 sur CPU — la **localité** décide de la vitesse des deux côtés.
"""

# ╔═╡ bf891e84-9d69-454a-a77b-724596266414
md"""
### Démonstration : le produit matriciel, CPU vs GPU

On déplace les matrices sur le GPU avec `cu(...)`, et on lance le **même** `*`. Le multiple dispatch (vu plus tôt !) choisit automatiquement la version GPU parce que les arguments sont maintenant des `CuArray`. Le même code source, du matériel différent.
"""

# ╔═╡ 78901ba1-a4f3-4808-99ed-b151bab95b78
if gpu_dispo
    let
        N = 4096
        Acpu = rand(Float32, N, N)
        Bcpu = rand(Float32, N, N)
        Agpu = cu(Acpu)              # copie vers la mémoire du GPU
        Bgpu = cu(Bcpu)

        print("CPU (Float32) : ")
        @btime $Acpu * $Bcpu

        print("GPU (Float32) : ")
        # CUDA.@sync attend la fin du calcul GPU (il est asynchrone !)
        @btime CUDA.@sync $Agpu * $Bgpu
    end
else
    md"*(démo GPU sautée — pas de carte détectée)*"
end

# ╔═╡ a3b77b02-29a2-455e-823d-d6068cfe3199
md"""
Sur une grande matrice, l'écart est important (souvent **10 à 50×**). Mais sur un **petit** problème, le GPU peut être **plus lent** que le CPU, à cause du coût de transfert et de lancement. **Le GPU n'est rentable qu'à grande échelle** — c'est l'objet de l'activité suivante.
"""

# ╔═╡ ba7c1100-0004-4a01-8b01-aaaaaaaa0004
md"""
### Le batching sur GPU : fournir assez de travail

Reprenons l'expérience du **débit par exemple** (section 3), mais en comparant **CPU et GPU** quand la taille du batch grandit. Une couche `W` (m → m) appliquée à `B` exemples empilés en colonnes (`W * X`, `X` de taille `m×B`).

Le GPU a des milliers de cœurs : à `B = 1` (un simple matrice-vecteur), la plupart restent inutilisés. Plus le batch grossit, plus le taux d'utilisation augmente.
"""

# ╔═╡ ba7c1100-0005-4a01-8b01-aaaaaaaa0005
if gpu_dispo
    let
        m = 4096                              # couche dense m → m, en Float32
        Wc = rand(Float32, m, m); Wg = cu(Wc)
        println("batch B | CPU µs/exemple | GPU µs/exemple | accélération GPU")
        println("-"^58)
        for B in (1, 8, 64, 256, 1024)
            Xc = rand(Float32, m, B); Xg = cu(Xc)
            tc = @belapsed $Wc * $Xc                  # CPU
            tg = @belapsed CUDA.@sync $Wg * $Xg       # GPU (asynchrone → @sync)
            println(@sprintf("%6d  | %13.3f | %14.3f | %10.1f×",
                             B, tc/B*1e6, tg/B*1e6, tc/tg))
        end
    end
else
    md"*(démo GPU sautée — pas de carte détectée ; la version CPU est en section 3)*"
end

# ╔═╡ ba7c1100-0006-4a01-8b01-aaaaaaaa0006
md"""
Sur GPU, le temps **par exemple** chute fortement quand `B` grandit : à `B = 1` le GPU est très largement sous-utilisé (souvent **plus lent** que le CPU), à grand `B` il dépasse nettement le CPU. C'est l'illustration la plus directe du fait qu'il faut fournir assez de travail au GPU — et la raison pour laquelle, en production, on infère toujours par **batchs**.
"""

# ╔═╡ 1ed9c688-150c-423c-a086-181603547e2c
md"""
## 5. Activité — quand le GPU devient-il rentable ?

On va tracer le facteur d'accélération GPU/CPU **en fonction de la taille** du problème, pour voir le point de bascule de nos propres yeux.
"""

# ╔═╡ 36c37805-567f-4baf-b8d3-5d42d5027b88
if gpu_dispo
    let
        println("taille N |  CPU (ms) |  GPU (ms) |  accélération")
        println("-"^48)
        for N in (64, 128, 256, 512, 1024, 2048, 4096)
            Ac = rand(Float32, N, N); Bc = rand(Float32, N, N)
            Ag = cu(Ac);              Bg = cu(Bc)
            tc = @belapsed $Ac * $Bc
            tg = @belapsed CUDA.@sync $Ag * $Bg
            println(@sprintf("%7d  | %8.3f  | %8.3f  |  %6.1f×", N, tc*1e3, tg*1e3, tc/tg))
        end
    end
else
    md"*(activité GPU sautée — pas de carte détectée)*"
end

# ╔═╡ 7663fb47-1f0c-4383-a5dd-e55f9a1b5abc
md"""
**Q1.** À partir de quelle taille `N` le GPU devient-il plus rapide que le CPU ? Pour les petites tailles, pourquoi est-il *plus lent* ? (Indice : transfert mémoire + lancement du *kernel*, coûts fixes indépendants de N.)

**Q2.** L'accélération continue-t-elle de croître avec N, ou plafonne-t-elle ? Que vous dit ce plafond sur la machine (pensez « bande passante » vs « calcul crête ») ?
"""

# ╔═╡ e85db6c9-b322-4a2f-aed6-295f21cb362d
md"""
## 6. Exemple conducteur : Monte-Carlo de π sur GPU

On a porté ce calcul du séquentiel au multi-thread CPU. Dernière étape : le GPU.

Le Monte-Carlo est **idéal** pour un GPU : des millions de tirages totalement indépendants. Mais souvenez-vous de la question Q4 du module sur les threads : **le générateur aléatoire doit fonctionner correctement avec des millions de threads simultanés.** `CUDA.jl` fournit exactement ça.

Plutôt que d'écrire un *kernel* explicite, on exprime le calcul de façon **vectorisée** : on tire tous les points directement sur le GPU et on compte ceux situés dans le quart de disque — toutes les données restent sur la carte.
"""

# ╔═╡ 7a6d400a-f826-4b67-9a7d-717351daa9bf
# Version GPU "vectorisée" : aucune boucle explicite, tout est parallélisé par CUDA.
function pi_gpu(n)
    x = CUDA.rand(Float32, n)            # n tirages, directement en mémoire GPU
    y = CUDA.rand(Float32, n)
    dans = (x.^2 .+ y.^2) .<= 1f0        # vecteur de Bool, calculé sur le GPU
    return 4 * sum(dans) / n             # sum réduit sur le GPU
end

# ╔═╡ aa59c041-e5fa-4cb0-a8f0-43c353255569
# Repli CPU équivalent, pour comparaison (et pour ceux sans GPU)
function pi_cpu_vec(n)
    x = rand(Float32, n)
    y = rand(Float32, n)
    dans = (x.^2 .+ y.^2) .<= 1f0
    return 4 * sum(dans) / n
end

# ╔═╡ 798cfc8b-51b7-4086-8b30-faa6be29c299
if gpu_dispo
    let
        n = 100_000_000
        println("π (référence) : ", π)
        println("pi_gpu : ", pi_gpu(n))
        print("CPU vectorisé : "); @btime pi_cpu_vec($n)
        print("GPU           : "); @btime CUDA.@sync pi_gpu($n)
    end
else
    let
        n = 20_000_000
        println("π (référence) : ", π)
        print("CPU vectorisé : "); @btime pi_cpu_vec($n)
        println("(pas de GPU : la version pi_gpu nécessite CUDA)")
    end
end

# ╔═╡ 0d3be671-416b-4c2f-a605-1995a602e132
md"""
**Q3.** Avec 100 millions de tirages, mesurez l'accélération GPU/CPU. Comparez aussi au gain qu'on avait obtenu avec les **threads CPU**. Le GPU vous donne-t-il bien plus que le nombre de cœurs CPU ? Pourquoi (combien de « threads » un GPU lance-t-il, en ordre de grandeur) ?

**Q4 (synthèse du cours).** Reprenez la loi d'Amdahl. Dans `pi_gpu`, qu'est-ce qui **reste séquentiel** ou coûteux et ne profite pas du parallélisme massif ? (Indice : l'allocation des vecteurs, la réduction finale `sum`, et — si on n'y prend pas garde — les transferts CPU↔GPU.) C'est ce qui empêche l'accélération d'être « infinie ».
"""

# ╔═╡ 130f0c20-5ca5-4045-9e70-e3898789bf01
md"""
## 7. Bilan — et synthèse du cours

### Ce qu'on a vu aujourd'hui

| Idée | Message |
|---|---|
| **Hiérarchie mémoire** | le CPU *attend* la mémoire ; les caches parient sur la localité |
| **Column-major (Julia)** | boucle intérieure sur le 1ᵉʳ indice ; l'inverse de numpy |
| **Intensité arithmétique** | peu d'ops/octet = *memory-bound* ; beaucoup = *compute-bound* |
| **Batching** | empiler les exemples : matrice-vecteur → matrice-matrice = memory-bound → compute-bound |
| **GPU** | des milliers de threads ; rentable seulement à grande échelle, suffisamment alimenté |

### Synthèse, en une phrase par module

1. **Compiler et spécialiser** supprime le surcoût *par opération* (Python interprété → Julia compilé).
2. **Le multiple dispatch** rend les fonctions à la fois génériques *et* rapides — et c'est lui qui fait basculer un `*` sur GPU sans changer le code.
3. **Paralléliser** exige de *minimiser le partage*, sinon le résultat est faux (la somme parallèle).
4. **Alimenter le matériel** : organiser les données pour respecter le cache et batcher, jusqu'à saturer le GPU.

### À retenir

> La performance ne vient pas du langage en soi. Elle vient de **comprendre ce que fait réellement la machine** : le coût de chaque opération, les données partagées, et leurs déplacements en mémoire. Julia permet d'observer tout cela — du code abaissé jusqu'à l'assembleur, de l'erreur de concurrence jusqu'au GPU. Ce sont des principes, pas des recettes : ils valent dans n'importe quel langage.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
"""

# ╔═╡ Cell order:
# ╟─aa07d25d-4f02-4013-95a0-49f72ddf7323
# ╟─ee0a05ee-935e-42c3-b567-5624fb902747
# ╠═3b62bc4c-9765-4424-b7ee-2800c085103a
# ╟─e60ad78b-4669-47b1-b819-443e4454f585
# ╟─cb000000-0000-4c00-9a00-00000000c001
# ╠═cb000000-0000-4c00-9a00-00000000c002
# ╠═cb000000-0000-4c00-9a00-00000000c003
# ╟─58cf2a8a-433f-4737-8116-044f7f8eada1
# ╠═3218651a-5b26-42d1-b330-917d0801ed44
# ╠═2126f361-6be9-4f14-a358-6a034213fa4f
# ╠═6908e575-a27a-43b8-b7ee-2cf0bce7f8fc
# ╠═43f403f1-a036-4454-8670-327d079dd5cc
# ╠═cdad600b-7259-4388-a83b-0a2f42f83aa1
# ╟─62969c4a-d889-4322-8923-725b34f839dc
# ╟─09844b2e-6544-494d-a7c1-50c3d5b9c5d6
# ╠═842ce51a-2b62-49e4-a447-1b842c6fe632
# ╠═01d52dcd-e327-418c-9906-dddd778b01f0
# ╟─ad57f7bf-d863-4be8-b5d4-d0dfd0603b69
# ╠═05884f27-4254-4a8e-8177-e38454dc8aac
# ╠═dc815f82-3a2e-40a7-b9a1-67daf6c38766
# ╟─fc30c6c4-6c52-4bad-b28e-5f54f21b4b1c
# ╟─ba7c1100-0001-4a01-8b01-aaaaaaaa0001
# ╠═ba7c1100-0002-4a01-8b01-aaaaaaaa0002
# ╟─ba7c1100-0003-4a01-8b01-aaaaaaaa0003
# ╟─92cd196c-2cab-4aee-aecb-89f06133486c
# ╟─cb000000-0000-4c00-9a00-00000000c004
# ╠═c3790242-0bd4-4168-858a-3e58d5bb3134
# ╠═e366e8e3-9f59-40bd-bb00-89347808fb1f
# ╟─cb000000-0000-4c00-9a00-00000000c005
# ╠═cb000000-0000-4c00-9a00-00000000c006
# ╟─cb000000-0000-4c00-9a00-00000000c007
# ╟─bf891e84-9d69-454a-a77b-724596266414
# ╠═78901ba1-a4f3-4808-99ed-b151bab95b78
# ╟─a3b77b02-29a2-455e-823d-d6068cfe3199
# ╟─ba7c1100-0004-4a01-8b01-aaaaaaaa0004
# ╠═ba7c1100-0005-4a01-8b01-aaaaaaaa0005
# ╟─ba7c1100-0006-4a01-8b01-aaaaaaaa0006
# ╟─1ed9c688-150c-423c-a086-181603547e2c
# ╠═36c37805-567f-4baf-b8d3-5d42d5027b88
# ╟─7663fb47-1f0c-4383-a5dd-e55f9a1b5abc
# ╟─e85db6c9-b322-4a2f-aed6-295f21cb362d
# ╠═7a6d400a-f826-4b67-9a7d-717351daa9bf
# ╠═aa59c041-e5fa-4cb0-a8f0-43c353255569
# ╠═798cfc8b-51b7-4086-8b30-faa6be29c299
# ╟─0d3be671-416b-4c2f-a605-1995a602e132
# ╟─130f0c20-5ca5-4045-9e70-e3898789bf01
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
