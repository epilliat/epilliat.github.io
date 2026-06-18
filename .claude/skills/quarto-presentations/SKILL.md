---
name: quarto-presentations
description: Authoring conventions for the Quarto reveal.js teaching decks (teaching/*/slides/*.qmd, research/presentations/*). Covers incremental pacing (`. . .` pauses), slide titles & the menu/index, LaTeX macros, structure, and per-directory `_metadata.yml`. For in-slide interactive JS/canvas demos use the separate `slide-animations` skill instead.
---

# Quarto reveal.js presentations — house conventions

These decks use `format: revealjs` with `incremental: true`, `css: ../../../styles.css`. Each
slides directory has a `_metadata.yml` (see bottom). Keep new decks consistent with the linear-model
and hypothesis-testing courses.

## Pacing: `. . .` pauses (the #1 thing people get wrong)

`. . .` on its own line is a Quarto **pause** (a reveal fragment break). With `incremental: true`:

- **Bullet/numbered lists pause by themselves** — every `-`/`1.` item is already a fragment, revealed
  one at a time. So **never put `. . .` directly before a top-level list** — it only adds an empty,
  content-less pause before the first bullet. (A list wrapped in `::: {.nonincremental}` does *not*
  auto-pause, so a `. . .` before *that* is fine.)
- **Pause after every slide title, before the first text.** A heading + a paragraph/`:::` block/math/
  table/image all appear *together* on arrival unless a `. . .` separates them. Put a `. . .` right
  after each `##` title so the **title lands alone first**, then the content reveals. (If the first
  content is a bare bullet list, you don't need it — the list already does this.)
- Use `. . .` between successive paragraphs/blocks you want revealed step by step.

Net rule per `##` slide: **title → `. . .` → first block**, *unless* the first block is a top-level
list (then no pause). A `. . .` immediately before `-`/`1.` is spurious — remove it.

```markdown
## Good fit                     ## Good list slide
                                
. . .                           - first point        ← list pauses on its own;
                                - second point          NO `. . .` before it
The residual sum of squares …   
```

`## ` lines **inside a `:::` block are not slides** — they are callout titles
(`::: {.callout-note}` / `## Theorem`). Never insert pauses or treat them as slide headings.

## Titles & the menu/index

- **Use Unicode in titles, not `$…$`** (`R²`, `β`, `σ̂²`, `Cₚ`, `χ²`, `Jₙ(β)`, `∈/∉/≤`). The reveal
  menu/index shows the raw title text; `$…$`/`\(…\)` render as raw TeX there. Math in the slide *body*
  stays normal `$…$`.
- Give every `#` section a clear name (it's the menu's table of contents) and **de-duplicate `##`
  titles** (e.g. two "Residuals" → "Residuals" / "Residuals: model vs estimation").
- End the deck with a `## Next` slide linking to the next deck, e.g.
  `[Introduction to the Generalized Linear Model](../../glm/slides/Introduction_glm.qmd)`.

## LaTeX macros

Put shared `\newcommand{…}` **after the first heading, inside a hidden div** — never before the first
heading (that makes an untitled orphan slide whose menu entry shows raw `\newcommand`). MathJax still
defines them (tex2jax processes `display:none`):

```markdown
# First section

::: {style="display:none"}
$\newcommand{\E}{\mathbb E}$ $\newcommand{\Var}{\mathbb V}$ …
:::
```

## Math delimiters by context

`$…$` in markdown prose (Pandoc converts it). Inside a raw ` ```{=html} ` block use `\(…\)`
(literal `$…$` there never typesets — the deck ships **MathJax 2.7.9**, inline delimiter `\(…\)` only).

## Per-directory `_metadata.yml`

```yaml
format:
  revealjs:
    history: false                 # browser Back leaves the deck, not steps slides
    include-after-body:
      text: |
        <script>                   # Alt+Left = browser Back (reveal otherwise eats it)
        window.addEventListener('keydown', function (e) {
          if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === 'ArrowLeft' || e.keyCode === 37)) {
            e.stopImmediatePropagation(); e.preventDefault(); if (history.length > 1) history.back();
          }
        }, true);
        </script>
```

## Verify

`quarto render teaching/<course>/slides/<deck>.qmd` → "Output created". Open at the preview server,
press **`o`** (overview) and the **☰** menu to check titles read cleanly. Interactive demos: see the
`slide-animations` skill (canvas, scaling, MathJax-2 typeset, persistence).
