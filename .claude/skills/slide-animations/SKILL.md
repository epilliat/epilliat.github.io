---
name: slide-animations
description: Build interactive JavaScript/canvas animations embedded in Quarto reveal.js teaching slides (e.g. teaching/linear_model/slides/*.qmd). Use when adding or editing an in-slide demo — draggable scatter plots, parameter sliders, distribution explorers, TeX-driven model equations. Encodes the reveal-specific gotchas (slide scaling, MathJax re-typeset, history) and the house style.
---

# Interactive slide animations (Quarto + reveal.js)

These decks are reveal.js (light theme, white background, `css: ../../../styles.css`). Animations are
plain `<canvas>` + a small IIFE inside a `` ```{=html} `` raw block on a `##` slide. No build step, no
dependencies. Keep them self-contained.

## Slide skeleton

```markdown
## Short title {.smaller}

::: {style="font-size:90%"}
One terse sentence — say what to look for, not how it's computed.
:::

```{=html}
<div id="xxWrap" style="text-align:center">
  <canvas id="xxCanvas" width="700" height="330" style="max-width:100%;background:#fff;border:1px solid #ddd;border-radius:8px;touch-action:none"></canvas>
  <div class="lm-row">
    <button class="lm-btn on" data-mode="a">A</button>
    <button class="lm-btn" data-mode="b">B</button>
    param <input id="xxP" type="range" min="0" max="1" step="0.01" value="0.5" style="vertical-align:middle;width:200px"> <b id="xxPv">0.50</b>
  </div>
</div>
<script>
(function () {
  var cv = document.getElementById('xxCanvas'); if (!cv || cv.dataset.init) return; cv.dataset.init = 1; // run once
  var c = cv.getContext('2d'), W = cv.width, H = cv.height;
  function gauss(){var u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  function draw(){ /* clear, map data->pixels, stroke/fill */ }
  // wire controls, then:
  draw();
})();
</script>
```
```

## Non-negotiable gotchas

1. **Reveal scales each slide.** A canvas declared `width=700` is *displayed* at some other size. Any
   mouse interaction MUST convert from displayed pixels to canvas pixels, or hit-tests miss:
   ```js
   function pos(e){var r=cv.getBoundingClientRect();return [(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height];}
   ```
   Forgetting this is the #1 bug ("drag does nothing"). It also covers the `max-width:100%` shrink.
2. **Fixed canvas size via `width`/`height` attributes** (not `offsetWidth`). Slides are `display:none`
   until visited, so layout-dependent sizing reads 0. Draw immediately on load; it shows when revealed.
3. **Run-once guard**: `if (cv.dataset.init) return; cv.dataset.init = 1;` — the inline script executes at
   page parse, and you don't want double-binding.
4. **Smooth morphing**: generate base random arrays ONCE (`u1/z1/z2/sign`, or `gX/gZ`) and recompute the
   points from slider params each `draw()`. Never re-sample on a slider drag (the cloud would jitter).
   A "↻ resample" button regenerates the base arrays.
5. **Dynamic TeX — this deck ships MathJax 2.7.9** (reveal's `plugin/math/math.js`), NOT MathJax 3.
   So `MathJax.typesetPromise` does **not exist** — using it leaves the raw source visible. Use the v2
   Hub API, and **retry until MathJax has loaded** (the inline script runs before MathJax):
   ```js
   function typesetEq(){ if (window.MathJax && MathJax.Hub && MathJax.Hub.Queue) MathJax.Hub.Queue(['Typeset', MathJax.Hub, eq]); else setTimeout(typesetEq, 200); }
   // on a mode switch: eq.innerHTML = '\\(' + tex + '\\)'; typesetEq();
   ```
   **Use `\(...\)` delimiters, NOT `$...$`.** The reveal MathJax config registers only
   `inlineMath: [['\\(','\\)']]` (Pandoc rewrites markdown `$...$`→`\(...\)`). A literal `$...$` written
   inside a raw `` ```{=html} `` block is passed through untouched and never typesets — the #1 cause of
   "I see the TeX code". Put `\(...\)` in the static div too. Only re-typeset on discrete changes (mode
   switch), not on every slider tick. Check the engine/delimiters first:
   `grep -oE "mathjax[^\"']*\.js|inlineMath[^]]*]]" docs/.../slide.html` — adapt if a deck uses MathJax 3 or KaTeX.
6. **Survive a page refresh** — three linked requirements:
   - **Persist** mode + parameters to `localStorage` on every change, and restore them on load (reveal
     fully reloads the page, so JS state is otherwise lost and resets to defaults):
     ```js
     var KEY = 'lmCorrExplorer';
     function save(){ try{ localStorage.setItem(KEY, JSON.stringify({s:shape, a:+p1.value, b:+p2.value})); }catch(e){} }
     var saved=null; try{ saved=JSON.parse(localStorage.getItem(KEY)); }catch(e){}
     setShape(saved && CFG[saved.s] ? saved.s : 'gauss');   // restore active mode/button
     if (saved){ p1.value=saved.a; if(p2w.style.display!=='none') p2.value=saved.b; }
     draw(); save();                                        // redraw + persist restored state
     ```
   - Add `autocomplete="off"` to every range `<input>` so the browser's own form-restoration doesn't
     fight the localStorage restore (they can disagree and cause a flash of the wrong value).
   - **Consistency**: after restoring, the active **button**, the slider positions, the **displayed
     numbers**, and the **equation** must all reflect the actual drawn state — achieve this by routing the
     restore through the same `setShape()` + `draw()` used at runtime (never set one without the other).
7. **A live equation must not reflow while dragging.** Re-typesetting a whole `\(...\)` on every
   `pointermove` makes the equation flicker and jump (async MathJax + changing token widths). Split it:
   typeset the **symbols once** (MathJax), and put every changing **number in its own fixed-width box**
   updated by `textContent` only — no MathJax during the drag. Re-typeset the symbols *only* on a discrete
   structural change (e.g. the active subscript `i,j` changing on `pointerdown`).
   ```html
   <span class="trm"><span class="sym" id="symA"></span><span class="val" id="valA">+0.00</span></span>
   ```
   ```css
   .trm{display:inline-flex;flex-direction:column;align-items:center;min-width:60px}
   .val{font-family:ui-monospace,monospace;width:56px;text-align:center}   /* fixed width ⇒ no reflow */
   ```
   ```js
   function renderSymbols(){ symA.innerHTML='\\(\\color{#2c6fb5}{\\alpha_'+(i+1)+'}\\)'; typeset(); } // on cell change only
   function updateNumbers(){ valA.textContent=sgn(alpha); valG.style.color=Math.abs(g)<.1?'#27ae60':'#c0392b'; } // every draw
   ```
   Colour cues (a value turning green at 0) are CSS `style.color` changes — instant, no reflow. Stack
   several boxed rows for a chained model (e.g. `Ŷ=μ+ε` above `μ=m+α+β+γ`). `\color{#RRGGBB}` works in
   MathJax 2 (unknown colour names pass straight through as CSS colours).

## House style

- **Light palette**: points `rgba(44,62,80,0.7)`, OLS/fit line `#e74c3c`, frame `#e6e6e6`, positive
  fill `rgba(52,152,219,.18)`, negative `rgba(231,76,60,.16)`, group colors `#3498db/#e67e22/#27ae60`.
- **Buttons** reuse the global `.lm-btn` (defined once, in the correlation slide). Keep them large:
  `font-size:.85em;padding:.4em 1.1em`. Active state: class `on` (blue fill).
- **Control row** `.lm-row { font-size:.72em }`. Sliders ~150–220px wide; show the value in a `<b>`.
- **Intro text**: one short line, `font-size:90%`. Simplify ruthlessly; the canvas carries the lesson.
- **Hatted symbols** (ρ̂, η̂², ĉov): in **HTML/DOM** write them as MathJax `\(\hat\rho\)` / `\(\hat\eta^2\)` /
  `\(\widehat{\mathrm{cov}}\)`, never the combining accent `&#770;` — it sits off-centre and looks broken.
  These static labels typeset on load; keep the *value* a plain `<span>` updated by JS next to the label.
  On a **`<canvas>`** there is no MathJax and the combining accent also renders wrong — **draw the caret
  yourself**: render the letter, then stroke a small chevron above it, e.g.
  `c.fillText('β',x,y); var w=c.measureText('β').width,cx=x+w/2; c.beginPath();c.moveTo(cx-3.5,y-12);c.lineTo(cx,y-16);c.lineTo(cx+3.5,y-12);c.stroke();`
  (Precomposed letters like `Ŷ` U+0176 are fine as-is, in DOM or canvas.)
- **Math delimiters by context**: in **markdown prose / intros** (a `:::` div, normal `.qmd` text) use
  `$…$` — Pandoc converts it. Inside a raw `` ```{=html} `` block (canvas labels are plain text/Unicode;
  any equation `<div>` is MathJax) use `\(…\)`. A literal `$…$` inside a raw HTML block never typesets.
- **Prefer an interpretable readout** over a raw formula: e.g. show $\cos(X_1,X_2)=\langle X_1,X_2\rangle/(\lVert X_1\rVert\lVert X_2\rVert)$
  (→1 at Cauchy–Schwarz equality) rather than `det(XᵀX)/(‖X₁‖²‖X₂‖²)`. Static `\(...\)` labels + dynamic `<span>` values.
- Clip overflow before drawing lines/points: `c.save();c.beginPath();c.rect(x,y,w,h);c.clip(); … c.restore();`.

## Interaction patterns

- **Slider**: `el.addEventListener('input', draw)`; read `+el.value` in `draw`.
- **Draggable handle/point**: pointer events + the scaled `pos()` above. `pointerdown` sets a drag index
  by nearest-within-radius; `pointermove` (guard `if(drag<0)return; …; e.preventDefault()`) updates and
  redraws; `pointerup` on `window` clears it. Need an inverse map `IY(py)` / `IX(px)`.
- **Mode buttons**: toggle `on` class, switch a `mode` var, update sliders' min/max/step/value + the TeX,
  then `draw()`.

## Deck-level requirements

- Each slides directory has a `_metadata.yml` that the deck inherits:
  ```yaml
  format:
    revealjs:
      history: false           # so browser Back leaves the deck instead of stepping slides
      include-after-body:
        text: |
          <script>             # Alt+Left = browser Back (reveal otherwise eats it)
          window.addEventListener('keydown', function (e) {
            if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === 'ArrowLeft' || e.keyCode === 37)) {
              e.stopImmediatePropagation(); e.preventDefault(); if (history.length > 1) history.back();
            }
          }, true);
          </script>
  ```
  A deck that defines its OWN `include-after-body` overrides this — add the Alt+Left script there too.

## Verify before publishing

```bash
quarto render teaching/linear_model/slides/introduction.qmd        # expect "Output created"
# extract each <script> containing your canvas id and node --check it:
python3 - docs/teaching/linear_model/slides/introduction.html <<'PY'
import re,sys,subprocess
h=open(sys.argv[1]).read()
for s in re.findall(r'<script>(.*?)</script>',h,re.S):
    if 'xxCanvas' in s:
        open('/tmp/a.js','w').write(s); print(subprocess.run(['node','--check','/tmp/a.js']).returncode)
PY
```
Sanity-check any statistics numerically in `node -e '…'` (e.g. compare a computed t-quantile to R).
Preview at the running server, e.g. `http://localhost:8099/teaching/linear_model/slides/introduction.html`,
hard-refresh, and actually drag/slide. Publishing renders to `docs/` and pushes `master` (Pages serves
`master/docs`); decks need `.nojekyll` in `docs/`.

## Existing examples to copy from

`teaching/linear_model/slides/introduction.qmd`: correlation explorer (modes + per-mode params + TeX),
covariance (drag points), least squares (drag line handles + residual squares), correlation test
(Student-t via Lanczos `lgamma`, rejection region, Monte-Carlo power), ANOVA (drag group means),
χ² (expected vs observed), overfitting (polynomial fit via normal equations).
