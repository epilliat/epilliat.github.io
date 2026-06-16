/* Self-contained, live replay player for the bandits deck.
 *
 * Replaces the old SVG-flip-book (which needed a 52 MB gitignored `replay/` symlink and
 * broke on the deployed site). Each ::: {.replay data-algo="<algo>"} ::: slide now
 * runs a small deterministic simulation of <algo> ∈ {ucb, ose, prose} (from data-algo) and draws
 * the SAME two-panel figure as the old frames:
 *
 *   left  — the arms (bars at the empirical mean, coloured by reservoir rank) with a [LCB,UCB]
 *           whisker; the exploration scope Z is shaded (OSE/PROSE); pulled arm = red border;
 *           recommended arm = gold ★.
 *   right — the reservoir curve A(1−xᵅ) with every generated arm as a dot, plus the noisy
 *           observation (red ✕). For UCB the right panel is a bar chart of the 5 true means.
 *
 * Controls (only on the current slide's replay): ← / → step a pull, ↓ / ↑ skip the slide.
 * Everything is deterministic (seeded), so a slide replays identically every time.
 */
(function () {
  "use strict";

  // ---- parameters (matched to ~/bandits/replay/gen_logs.jl) ---------------
  var A = 2.0, ALPHA = 1.5, BETA = 2.0, GAMMA = 0.0, SIGMA = 0.5, TMAX = 100;
  var K_RES = 40, K_UCB = 5, MLO = 0.55, MHI = 0.90;

  // ---- seeded PRNG + gaussian --------------------------------------------
  var rng = Math.random;
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss() { var u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // viridis-ish colormap (rank 0 = best/purple … 1 = worst/yellow)
  var VIR = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
  function viridis(t) { t = Math.max(0, Math.min(1, t)) * (VIR.length - 1); var i = Math.floor(t), f = t - i, a = VIR[i], b = VIR[Math.min(VIR.length - 1, i + 1)];
    return "rgb(" + Math.round(a[0] + (b[0] - a[0]) * f) + "," + Math.round(a[1] + (b[1] - a[1]) * f) + "," + Math.round(a[2] + (b[2] - a[2]) * f) + ")"; }

  // ---- per-run reservoir + shared observation noise ----------------------
  var MEANS, RANKX, NOISE;
  function noiseFor(a, k) { var arr = NOISE[a]; while (arr.length <= k) arr.push(gauss()); return arr[k]; }
  function newStats(K) { var s = []; for (var i = 0; i < K; i++) s.push({ pulls: 0, sum: 0 }); return s; }
  function randperm(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); for (var j = n - 1; j > 0; j--) { var k = (rng() * (j + 1)) | 0, t = a[j]; a[j] = a[k]; a[k] = t; } return a; }

  // ---- algorithms (deterministic; one pull per step) ---------------------
  function emp(st, i) { var s = st[i]; return s.pulls ? s.sum / s.pulls : NaN; }
  function hw(st, i) { var s = st[i]; return s.pulls ? Math.sqrt(BETA / s.pulls) : Infinity; }
  function pull(st, i) { var s = st[i], r = MEANS[i] + SIGMA * noiseFor(i, s.pulls); s.pulls++; s.sum += r; return r; }
  function row(st, i) { var e = emp(st, i), h = hw(st, i); return { idx: i, emp: e, lcb: isNaN(e) ? -Infinity : e - h, ucb: isNaN(e) ? Infinity : e + h, pulls: st[i].pulls }; }

  function Scope(K, jmaxCeil) { this.K = K; this.st = newStats(K); this.J = 1; this.t = 0; this.explored = []; this.order = randperm(K);
    this.scope = 0; this.pulled = -1; this.reco = -1; this.obs = NaN; this.jmaxCeil = jmaxCeil; }
  Scope.prototype.zForT = function (t) { var Jmax = this.jmaxCeil ? Math.ceil(Math.log2(t + 1)) : Math.floor(Math.log2(t + 1)); if (Jmax < 1) Jmax = 1;
    if (this.J > Jmax) this.J = 1; var U = Math.pow(this.J / Jmax, Math.exp(-GAMMA)); return Math.floor(Math.pow(t, U)); };
  Scope.prototype.step = function (sortByLcb) {
    this.t++; var t = this.t, Z = this.zForT(t), st = this.st, self = this;
    if (Z > this.explored.length && this.explored.length < this.K) this.explored.push(this.order[this.explored.length]);
    if (sortByLcb) this.explored.sort(function (a, b) { return (row(st, b).lcb - row(st, a).lcb) || (a - b); });
    Z = Math.min(Z, this.explored.length); this.scope = Z;
    var bi = 0; for (var i = 1; i < Z; i++) { var e = this.explored[i], be = this.explored[bi]; var ue = row(st, e).ucb, ub = row(st, be).ucb; if (ue > ub || (ue === ub && e < be)) bi = i; }
    var idx = this.explored[bi]; this.pulled = idx; this.obs = pull(st, idx);
    if (sortByLcb) { this.explored.sort(function (a, b) { return (row(st, b).lcb - row(st, a).lcb) || (a - b); }); this.reco = this.explored[0]; }
    else { var r = 0; for (var j = 1; j < this.explored.length; j++) { var ej = this.explored[j], br = this.explored[r]; var lj = row(st, ej).lcb, lr = row(st, br).lcb; if (lj > lr || (lj === lr && ej < br)) r = j; } this.reco = this.explored[r]; }
    this.J++;
  };
  Scope.prototype.snapshot = function (sortByLcb) {
    var st = this.st, ids = this.explored.slice();
    var rows = ids.map(function (i) { return row(st, i); });
    return { rows: rows, scope: this.scope, pulled: this.pulled, reco: this.reco, obs: this.obs, seen: ids.slice() };
  };

  function UCB(K) { this.K = K; this.st = newStats(K); this.t = 0; this.pulled = -1; this.reco = -1; this.obs = NaN; }
  UCB.prototype.step = function () { var st = this.st, K = this.K; this.t++;
    var bi = 0; for (var i = 1; i < K; i++) { var ui = row(st, i).ucb, ub = row(st, bi).ucb; if (ui > ub || (ui === ub && i < bi)) bi = i; }
    this.pulled = bi; this.obs = pull(st, bi);
    var r = 0; for (var j = 1; j < K; j++) { var lj = row(st, j).lcb, lr = row(st, r).lcb; if (lj > lr || (lj === lr && j < r)) r = j; } this.reco = r; };
  UCB.prototype.snapshot = function () { var st = this.st, rows = []; for (var i = 0; i < this.K; i++) rows.push(row(st, i));
    return { rows: rows, scope: 0, pulled: this.pulled, reco: this.reco, obs: this.obs, seen: rows.map(function (r) { return r.idx; }) }; };

  // ---- build one replay: generate arms, run TMAX steps, cache snapshots ---
  function build(algo) {
    var seed = ({ ucb: 11, ose: 23, prose: 41 })[algo] || 7;
    rng = mulberry32(seed);
    var K = algo === "ucb" ? K_UCB : K_RES;
    MEANS = []; RANKX = []; NOISE = [];
    for (var i = 0; i < K; i++) {
      var g, m;
      if (algo === "ucb") { m = A * (MLO + (MHI - MLO) * rng()); g = Math.pow(Math.max(0, Math.min(1, 1 - m / A)), 1 / ALPHA); }
      else { g = rng(); m = A * (1 - Math.pow(g, ALPHA)); }
      MEANS.push(m); RANKX.push(g); NOISE.push([]);
    }
    var sim = algo === "ucb" ? new UCB(K) : new Scope(K, algo === "prose");
    var byLcb = algo === "prose";
    var snaps = [algo === "ucb" ? sim.snapshot() : sim.snapshot(byLcb)];
    for (var t = 1; t <= TMAX; t++) { algo === "ucb" ? sim.step() : sim.step(byLcb); snaps.push(algo === "ucb" ? sim.snapshot() : sim.snapshot(byLcb)); }
    // stable top of left y-axis (≈98th pct of finite UCBs)
    var us = []; snaps.forEach(function (s) { s.rows.forEach(function (r) { if (isFinite(r.ucb)) us.push(r.ucb); }); });
    us.sort(function (a, b) { return a - b; });
    var p98 = us.length ? us[Math.min(us.length - 1, Math.floor(0.98 * us.length))] : 1.5 * A;
    var yhi = Math.max(1.3 * A, Math.min(3 * A, 1.08 * p98));
    return { algo: algo, K: K, mean: MEANS.slice(), rank: RANKX.slice(), snaps: snaps, yhi: yhi };
  }

  // ---- drawing: two panels reproducing the old frames --------------------
  function draw(r) {
    var cv = r.canvas, dpr = window.devicePixelRatio || 1, w = cv.offsetWidth, h = cv.offsetHeight;
    if (!w || !h) return;
    cv.width = w * dpr; cv.height = h * dpr; var c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h); c.fillStyle = "#ffffff"; c.fillRect(0, 0, w, h);   // white figure (matches the old frames)
    c.font = "11px system-ui,sans-serif";
    var snap = r.snaps[r.cur], algo = r.algo, K = r.K, yhi = r.yhi, isUcb = algo === "ucb";

    // layout: left ~57%, right ~43%
    var Tm = 12, Bm = 26, splitL = Math.round(w * 0.57);
    var Lx0 = 42, Lx1 = splitL - 14, Rx0 = splitL + 40, Rx1 = w - 14;
    var plotH = h - Tm - Bm;
    function LY(v) { return Tm + (1 - Math.max(0, Math.min(1, v / yhi))) * plotH; }
    var slotW = (Lx1 - Lx0) / K, LX = function (p) { return Lx0 + (p + 0.5) * slotW; };

    // ---- LEFT panel ----
    // scope shading (OSE/PROSE)
    if (!isUcb && snap.scope > 0) {
      var edge = LX(Math.min(snap.scope, K) - 1) + slotW * 0.5;
      c.fillStyle = "rgba(70,120,230,0.13)"; c.fillRect(Lx0 - 4, Tm, edge - (Lx0 - 4), plotH);
      c.strokeStyle = "rgba(70,120,230,0.85)"; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(edge, Tm); c.lineTo(edge, Tm + plotH); c.stroke(); c.setLineDash([]);
      c.fillStyle = "rgba(45,95,205,0.95)"; c.fillText("scope Z=" + snap.scope, Math.min(edge + 4, Lx1 - 60), Tm + 11);
    }
    // y axis + A line
    c.strokeStyle = "rgba(0,0,0,0.22)"; c.beginPath(); c.moveTo(Lx0, Tm); c.lineTo(Lx0, Tm + plotH); c.lineTo(Lx1, Tm + plotH); c.stroke();
    c.strokeStyle = "rgba(0,0,0,0.28)"; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(Lx0, LY(A)); c.lineTo(Lx1, LY(A)); c.stroke(); c.setLineDash([]);
    c.fillStyle = "rgba(55,60,74,0.9)"; c.save(); c.translate(12, Tm + plotH / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText("empirical mean · [LCB, UCB]", 0, 0); c.restore();

    snap.rows.forEach(function (rw, p) {
      var x = LX(p), col = viridis(r.rank[rw.idx]), isPull = rw.idx === snap.pulled, isReco = rw.idx === snap.reco, bw = slotW * 0.82;
      if (isNaN(rw.emp)) { // unexplored: faint full-height bar + arrows
        c.globalAlpha = 0.2; c.fillStyle = col; c.fillRect(x - bw / 2, Tm, bw, plotH); c.globalAlpha = 1;
        if (isPull) { c.strokeStyle = "#ff5252"; c.lineWidth = 2; c.strokeRect(x - bw / 2, Tm, bw, plotH); }
        if (isReco) star(c, x, isUcb ? Tm : Tm + plotH);
        return;
      }
      // bar at empirical mean
      var by = LY(rw.emp); c.fillStyle = col; c.fillRect(x - bw / 2, by, bw, Tm + plotH - by);
      c.strokeStyle = isPull ? "#ff5252" : "rgba(0,0,0,0.40)"; c.lineWidth = isPull ? 2 : 1; c.strokeRect(x - bw / 2, by, bw, Tm + plotH - by);
      // whisker [lcb,ucb]
      var hi = isFinite(rw.ucb) ? rw.ucb : yhi, lo = Math.max(rw.lcb, 0), cap = Math.min(slotW * 0.3, 6);
      c.strokeStyle = "rgba(20,24,34,0.95)"; c.lineWidth = 1.4;
      c.beginPath(); c.moveTo(x, LY(Math.min(hi, yhi))); c.lineTo(x, LY(lo)); c.stroke();
      c.beginPath(); c.moveTo(x - cap, LY(lo)); c.lineTo(x + cap, LY(lo));
      if (rw.ucb <= yhi) { c.moveTo(x - cap, LY(hi)); c.lineTo(x + cap, LY(hi)); } c.stroke();
      if (rw.ucb > yhi) { c.fillStyle = "rgba(20,24,34,0.95)"; tri(c, x, Tm, true); }
      if (isReco) star(c, x, LY(isUcb ? Math.min(rw.ucb, yhi) : lo));
    });
    c.fillStyle = "rgba(55,60,74,0.9)"; c.textAlign = "center";
    c.fillText(isUcb ? "arms (fixed order)" : algo === "ose" ? "arms (arrival order · scope = first Z)" : "arms (sorted by LCB)", (Lx0 + Lx1) / 2, h - 9);
    c.textAlign = "left";

    // ---- RIGHT panel ----
    var rylo = -0.12, ryhi = A * 1.12;
    function RYv(v) { return Tm + (1 - (v - rylo) / (ryhi - rylo)) * plotH; }
    c.strokeStyle = "rgba(0,0,0,0.22)"; c.beginPath(); c.moveTo(Rx0, Tm); c.lineTo(Rx0, Tm + plotH); c.lineTo(Rx1, Tm + plotH); c.stroke();
    if (isUcb) {
      // bar chart of the K true means, best (highest) first
      var ord = r.mean.map(function (_, i) { return i; }).sort(function (a, b) { return r.mean[b] - r.mean[a]; });
      var sW = (Rx1 - Rx0) / r.K, RX = function (p) { return Rx0 + (p + 0.5) * sW; };
      ord.forEach(function (idx, p) { var x = RX(p), col = viridis(r.rank[idx]), by = RYv(r.mean[idx]);
        c.fillStyle = col; c.fillRect(x - sW * 0.41, by, sW * 0.82, RYv(0) - by);
        c.strokeStyle = idx === snap.pulled ? "#ff5252" : "rgba(0,0,0,0.40)"; c.lineWidth = idx === snap.pulled ? 2 : 1; c.strokeRect(x - sW * 0.41, by, sW * 0.82, RYv(0) - by);
        if (idx === snap.reco) star(c, x, by - 2);
        if (idx === snap.pulled && isFinite(snap.obs)) { obsX(c, x, RYv(r.mean[idx]), RYv(snap.obs)); }
      });
      c.fillStyle = "rgba(55,60,74,0.9)"; c.textAlign = "center"; c.fillText(r.K + " arms — true means", (Rx0 + Rx1) / 2, h - 9); c.textAlign = "left";
    } else {
      // reservoir curve + observed arms as dots + observation ✕
      function RX(g) { return Rx0 + Math.max(0, Math.min(1, g)) * (Rx1 - Rx0); }
      c.strokeStyle = "rgba(70,74,88,0.7)"; c.lineWidth = 1.6; c.beginPath();
      for (var s = 0; s <= 100; s++) { var xx = s / 100, px = RX(xx), py = RYv(A * (1 - Math.pow(xx, ALPHA))); s ? c.lineTo(px, py) : c.moveTo(px, py); } c.stroke();
      snap.seen.forEach(function (i) { c.fillStyle = viridis(r.rank[i]); c.beginPath(); c.arc(RX(r.rank[i]), RYv(r.mean[i]), 3.4, 0, 6.2832); c.fill();
        c.strokeStyle = "rgba(0,0,0,0.5)"; c.lineWidth = 0.6; c.stroke(); });
      if (snap.pulled >= 0) { c.strokeStyle = "#ff5252"; c.lineWidth = 2; c.beginPath(); c.arc(RX(r.rank[snap.pulled]), RYv(r.mean[snap.pulled]), 5.5, 0, 6.2832); c.stroke();
        if (isFinite(snap.obs)) obsX(c, RX(r.rank[snap.pulled]), RYv(r.mean[snap.pulled]), RYv(snap.obs)); }
      if (snap.reco >= 0) star(c, RX(r.rank[snap.reco]), RYv(r.mean[snap.reco]));
      c.fillStyle = "rgba(55,60,74,0.9)"; c.textAlign = "center"; c.fillText("reservoir: " + snap.seen.length + "/" + K + " arms · A(1−xᵅ)", (Rx0 + Rx1) / 2, h - 9); c.textAlign = "left";
    }

    // caption
    if (r.caption) r.caption.textContent = "t = " + r.cur + " / " + TMAX +
      (snap.pulled >= 0 && r.cur > 0 ? "   ·   pull arm " + snap.pulled + "   ·   recommend arm " + snap.reco + " ★" : "   ·   (press → to start)");
  }
  function star(c, x, y) { c.fillStyle = "#f5c518"; c.font = "16px system-ui,sans-serif"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("★", x, y); c.textBaseline = "alphabetic"; c.textAlign = "left"; c.font = "11px system-ui,sans-serif"; }
  function obsX(c, x, y0, y1) { c.strokeStyle = "rgba(255,82,82,0.6)"; c.lineWidth = 1; c.setLineDash([2, 2]); c.beginPath(); c.moveTo(x, y0); c.lineTo(x, y1); c.stroke(); c.setLineDash([]);
    c.strokeStyle = "#ff5252"; c.lineWidth = 2; var s = 3.5; c.beginPath(); c.moveTo(x - s, y1 - s); c.lineTo(x + s, y1 + s); c.moveTo(x + s, y1 - s); c.lineTo(x - s, y1 + s); c.stroke(); }
  function tri(c, x, y, up) { c.beginPath(); c.moveTo(x - 4, up ? y + 5 : y - 5); c.lineTo(x + 4, up ? y + 5 : y - 5); c.lineTo(x, y); c.closePath(); c.fill(); }

  // ---- build each .replay element ----------------------------------------
  function algoOf(el) { var a = (el.getAttribute("data-algo") || "").toLowerCase(); return /^(ucb|ose|prose)$/.test(a) ? a : "ose"; }
  function setup(el) {
    var data = build(algoOf(el));
    var cv = document.createElement("canvas"); cv.className = "replay-canvas";
    var cap = document.createElement("div"); cap.className = "replay-caption";
    el.appendChild(cv); el.appendChild(cap);
    var r = { algo: data.algo, K: data.K, mean: data.mean, rank: data.rank, snaps: data.snaps, yhi: data.yhi, canvas: cv, caption: cap, cur: 0, max: data.snaps.length - 1 };
    el.__replay = r; draw(r);
    return r;
  }

  // ---- reveal.js integration (same control scheme as the old player) ------
  function currentReplay(reveal) { var slide = reveal.getCurrentSlide(); if (!slide) return null; var el = slide.querySelector(".replay"); return el && el.__replay ? el.__replay : null; }
  function init(reveal) {
    var els = document.querySelectorAll(".replay"); for (var i = 0; i < els.length; i++) setup(els[i]);
    document.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return; var r = currentReplay(reveal); if (!r) return;
      switch (e.key) {
        case "ArrowRight": if (r.cur < r.max) { r.cur++; draw(r); e.preventDefault(); e.stopImmediatePropagation(); } break;
        case "ArrowLeft":  if (r.cur > 0) { r.cur--; draw(r); e.preventDefault(); e.stopImmediatePropagation(); } break;
        case "ArrowDown":  e.preventDefault(); e.stopImmediatePropagation(); reveal.next(); break;
        case "ArrowUp":    e.preventDefault(); e.stopImmediatePropagation(); reveal.prev(); break;
      }
    }, true);
    if (reveal.on) reveal.on("slidechanged", function () { var r = currentReplay(reveal); if (r) draw(r); });
    window.addEventListener("resize", function () { var r = currentReplay(reveal); if (r) draw(r); });
  }
  function waitForReveal() {
    if (window.Reveal && window.Reveal.isReady && window.Reveal.isReady()) init(window.Reveal);
    else if (window.Reveal && window.Reveal.on) window.Reveal.on("ready", function () { init(window.Reveal); });
    else setTimeout(waitForReveal, 80);
  }
  waitForReveal();
})();
