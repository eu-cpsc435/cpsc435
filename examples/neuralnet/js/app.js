/* Rendering and controls. The maths lives in nn.js. */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var POS = [42, 120, 214], NEG = [208, 59, 59], WHITE = [255, 255, 255];
  var NEUTRAL = '#c9c8c3';
  var MAX = { inputs: 4, outputs: 3, layers: 3, neurons: 5 };
  var SPEEDS = [2, 20, 200, 2000];           // steps per second

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    sizes: [2, 1],
    act: 'sigmoid',
    task: 'AND',
    seed: 1,
    lr: 1,
    mode: 'point',                           // 'point' = loss on the current point, 'all' = mean over the data
    point: 0,
    selNode: { l: 1, j: 0 },
    axes: null,                              // the two parameters shown on the loss surface
    nextAxis: 0,
    range: 4,
    trail: [],
    hist: [], histStride: 1, steps: 0,
    running: false,
    yaw: -0.65, pitch: 0.62
  };
  var net, data;
  var graphs = { fwd: null, bwd: null };

  // ------------------------------------------------------------- helpers
  function fmt(v, d) {
    var s = Math.abs(v).toFixed(d === undefined ? 2 : d);
    return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
  }
  function mix(c, t) {
    t = Math.max(0, Math.min(1, t));
    return 'rgb(' + c.map(function (v, k) { return Math.round(WHITE[k] + (v - WHITE[k]) * t); }).join(',') + ')';
  }
  function rgb(c) { return 'rgb(' + c.join(',') + ')'; }
  function el(name, attrs, parent) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function key(p) { return p.l + '_' + p.j + '_' + p.i; }
  function same(p, q) { return p.l === q.l && p.j === q.j && p.i === q.i; }
  function lossPoints() { return state.mode === 'point' ? [data[state.point]] : data; }

  // ------------------------------------------------------- model lifecycle
  function rebuild(newData) {
    net = NN.create(state.sizes, state.act, state.seed);
    var nIn = state.sizes[0], nOut = state.sizes[state.sizes.length - 1];
    if (newData || !data) data = NN.makeData(nIn, nOut, state.task);
    // start on the all-ones point: with an input of 0 its weight has no effect on the loss
    if (newData) state.point = data.length - 1;

    var last = net.W.length - 1;
    state.selNode = { l: last + 1, j: 0 };
    state.axes = [{ l: last, j: 0, i: 0 }, { l: last, j: 0, i: state.sizes[last] > 1 ? 1 : -1 }];
    state.nextAxis = 0;
    resetRun();
    buildGraph('fwd');
    buildGraph('bwd');
    buildTable();
    render();
  }

  function resetRun() {
    state.trail = [];
    state.range = 4;
    state.steps = 0;
    state.hist = [NN.loss(net, data)];
    state.histStride = 1;
  }

  function trainStep() {
    var g = NN.gradient(net, lossPoints());
    NN.step(net, g, state.lr);
    state.steps++;
    if (state.mode === 'point') state.point = (state.point + 1) % data.length;

    state.trail.push([NN.getParam(net, state.axes[0]), NN.getParam(net, state.axes[1])]);
    if (state.trail.length > 400) state.trail.shift();
    if (state.steps % state.histStride === 0) {
      state.hist.push(NN.loss(net, data));
      if (state.hist.length > 1200) {
        state.hist = state.hist.filter(function (_, k) { return k % 2 === 0; });
        state.histStride *= 2;
      }
    }
  }

  // ------------------------------------------------------------ the graph
  function layout() {
    var n = state.sizes.length, pos = [];
    for (var l = 0; l < n; l++) {
      var x = n === 1 ? 60 : 60 + 380 * l / (n - 1), col = [], m = state.sizes[l];
      for (var j = 0; j < m; j++) col.push({ x: x, y: 124 + (j - (m - 1) / 2) * 50 });
      pos.push(col);
    }
    return pos;
  }
  var TARGET_X = 536, R = 17;

  function buildGraph(kind) {
    var svg = $(kind);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var pos = layout(), G = { pos: pos, edges: {}, nodes: [], targets: [], tags: [] };
    var gEdges = el('g', {}, svg), gNodes = el('g', {}, svg), gTags = el('g', {}, svg);

    function addEdge(p, x1, y1, x2, y2) {
      var hit = el('line', { 'class': 'hit', x1: x1, y1: y1, x2: x2, y2: y2 }, gEdges);
      var line = el('line', { 'class': 'edge', x1: x1, y1: y1, x2: x2, y2: y2, 'pointer-events': 'none' }, gEdges);
      var title = el('title', {}, hit);
      hit.addEventListener('click', function () { pickAxis(p); });
      // tag sits near the source end so tags on crossing edges do not collide
      G.edges[key(p)] = { line: line, title: title, tx: x1 + (x2 - x1) * 0.3, ty: y1 + (y2 - y1) * 0.3 };
    }

    for (var l = 0; l < net.W.length; l++) {
      for (var j = 0; j < net.W[l].length; j++) {
        var to = pos[l + 1][j];
        for (var i = 0; i < net.W[l][j].length; i++) {
          var from = pos[l][i];
          addEdge({ l: l, j: j, i: i }, from.x, from.y, to.x, to.y);
        }
        // bias: a short stub entering the neuron from the upper left
        addEdge({ l: l, j: j, i: -1 }, to.x - 27, to.y - 25, to.x - 11, to.y - 10);
        el('circle', { cx: to.x - 27, cy: to.y - 25, r: 2.5, fill: '#8a8983', 'pointer-events': 'none' }, gEdges);
      }
    }

    pos.forEach(function (col, l) {
      G.nodes.push(col.map(function (pt, j) {
        var g = el('g', { 'class': 'node' + (l > 0 && kind === 'fwd' ? ' click' : '') }, gNodes);
        var c = el('circle', { cx: pt.x, cy: pt.y, r: R }, g);
        var t = el('text', { x: pt.x, y: pt.y }, g);
        var title = el('title', {}, g);
        if (l > 0 && kind === 'fwd') g.addEventListener('click', function () { state.selNode = { l: l, j: j }; render(); });
        return { g: g, c: c, t: t, title: title };
      }));
    });

    // right-hand column: the target (forward) or the error (backward)
    var outs = pos[pos.length - 1];
    outs.forEach(function (pt) {
      el('line', { 'class': 'err', x1: pt.x + R + 4, y1: pt.y, x2: TARGET_X - R - 4, y2: pt.y }, gEdges);
      if (kind === 'fwd') {
        var g = el('g', { 'class': 'node target' }, gNodes);
        el('circle', { cx: TARGET_X, cy: pt.y, r: R }, g);
        G.targets.push(el('text', { x: TARGET_X, y: pt.y }, g));
      } else {
        G.targets.push(el('text', { 'class': 'errtxt', x: TARGET_X, y: pt.y }, gNodes));
      }
    });

    var labelY = 272;
    function label(x, text) { el('text', { 'class': 'lbl', x: x, y: labelY }, svg).textContent = text; }
    function sbtn(x, sign, enabled, fn) {
      var g = el('g', { 'class': 'sbtn' + (enabled ? '' : ' off'), role: 'button', tabindex: enabled ? 0 : -1,
                        'aria-label': (sign > 0 ? 'Add' : 'Remove') }, svg);
      el('circle', { cx: x, cy: labelY, r: 9 }, g);
      el('text', { x: x, y: labelY }, g).textContent = sign > 0 ? '+' : '−';
      g.addEventListener('click', fn);
      g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } });
    }
    if (kind === 'fwd') {
      pos.forEach(function (col, l) {
        var x = col[0].x, lastL = pos.length - 1;
        var name = l === 0 ? 'inputs' : l === lastL ? 'outputs' : 'hidden';
        var max = l === 0 ? MAX.inputs : l === lastL ? MAX.outputs : MAX.neurons;
        label(x, name);
        sbtn(x - 36, -1, state.sizes[l] > 1, function () { resize(l, -1); });
        sbtn(x + 36, +1, state.sizes[l] < max, function () { resize(l, +1); });
      });
      label(TARGET_X, 'target');
    } else {
      label(TARGET_X, 'error');
    }

    ['A', 'B'].forEach(function (name) {
      var g = el('g', { 'class': 'axtag', 'pointer-events': 'none' }, gTags);
      el('rect', { x: -8, y: -8, width: 16, height: 16, rx: 4 }, g);
      el('text', { x: 0, y: 0.5 }, g).textContent = name;
      G.tags.push(g);
    });
    graphs[kind] = G;
  }

  function resize(l, d) {
    state.sizes[l] += d;
    var ioChanged = l === 0 || l === state.sizes.length - 1;
    rebuild(ioChanged);
  }

  function pickAxis(p) {
    if (same(p, state.axes[0]) || same(p, state.axes[1])) return;
    state.axes[state.nextAxis] = p;
    state.nextAxis = 1 - state.nextAxis;
    state.trail = [];
    state.range = 4;
    render();
  }

  function updateGraphs(fwd, g) {
    var F = graphs.fwd, B = graphs.bwd, l, j, i;
    var maxG = 1e-9, maxD = 1e-9;
    eachParam(function (p) { maxG = Math.max(maxG, Math.abs(NN.getGrad(g, p))); });
    g.delta.forEach(function (row) { row.forEach(function (d) { maxD = Math.max(maxD, Math.abs(d)); }); });

    eachParam(function (p) {
      var k = key(p), w = NN.getParam(net, p), gr = NN.getGrad(g, p);
      var name = p.i < 0 ? 'bias' : 'weight';
      var tip = name + ' = ' + fmt(w, 3) + '\n∂L/∂' + (p.i < 0 ? 'b' : 'w') + ' = ' + fmt(gr, 4);

      var fe = F.edges[k];
      fe.line.setAttribute('stroke', w >= 0 ? rgb(POS) : rgb(NEG));
      fe.line.setAttribute('stroke-width', 0.8 + Math.min(Math.abs(w), 6) * 1.1);
      fe.line.setAttribute('stroke-opacity', 0.85);
      fe.title.textContent = tip;

      var be = B.edges[k], rel = Math.abs(gr) / maxG, flat = maxG < 1e-7 || rel < 0.02;
      // descent moves a weight opposite to its gradient
      be.line.setAttribute('stroke', flat ? NEUTRAL : gr < 0 ? rgb(POS) : rgb(NEG));
      be.line.setAttribute('stroke-width', flat ? 1 : 1 + rel * 6);
      be.line.setAttribute('stroke-opacity', 0.9);
      be.title.textContent = tip;
    });

    for (l = 0; l < state.sizes.length; l++) {
      for (j = 0; j < state.sizes[l]; j++) {
        var a = fwd.a[l][j], fn = F.nodes[l][j], bn = B.nodes[l][j];
        fn.c.setAttribute('fill', mix(a >= 0 ? POS : NEG, Math.abs(a) * 0.85));
        fn.t.textContent = l === 0 ? String(a) : fmt(a);
        fn.t.setAttribute('fill', Math.abs(a) > 0.62 ? '#fff' : '#1f1f1e');
        fn.title.textContent = l === 0 ? 'input = ' + a : 'z = ' + fmt(fwd.z[l][j], 3) + '\nactivation = ' + fmt(a, 3);
        fn.g.classList.toggle('sel', l === state.selNode.l && j === state.selNode.j);

        if (l === 0) {
          bn.c.setAttribute('fill', '#f1f0ec');
          bn.t.textContent = '';
        } else {
          var d = g.delta[l - 1][j], t = Math.abs(d) / maxD;
          bn.c.setAttribute('fill', maxD < 1e-7 ? '#fff' : mix(d < 0 ? POS : NEG, t * 0.85));
          bn.t.textContent = fmt(d);
          bn.t.setAttribute('fill', t > 0.62 && maxD >= 1e-7 ? '#fff' : '#1f1f1e');
          bn.title.textContent = '∂L/∂z = ' + fmt(d, 4);
        }
      }
    }

    var out = fwd.a[fwd.a.length - 1], y = data[state.point].y;
    for (j = 0; j < out.length; j++) {
      F.targets[j].textContent = String(y[j]);
      B.targets[j].textContent = state.mode === 'point' ? fmt(out[j] - y[j]) : '';
    }

    [F, B].forEach(function (G) {
      state.axes.forEach(function (p, n) {
        var e = G.edges[key(p)];
        G.tags[n].setAttribute('transform', 'translate(' + e.tx + ',' + e.ty + ')');
      });
    });
  }

  function eachParam(fn) {
    for (var l = 0; l < net.W.length; l++)
      for (var j = 0; j < net.W[l].length; j++) {
        for (var i = 0; i < net.W[l][j].length; i++) fn({ l: l, j: j, i: i });
        fn({ l: l, j: j, i: -1 });
      }
  }

  // ----------------------------------------------------------- math lines
  function updateMath(fwd, g) {
    var n = state.selNode, l = n.l - 1, isOut = n.l === state.sizes.length - 1;
    var f = isOut ? 'σ' : NN.ACT[state.act].label;
    var terms = '';
    net.W[l][n.j].forEach(function (w, i) {
      var x = fwd.a[l][i], xs = l === 0 ? String(x) : fmt(x);
      terms += (i === 0 ? (w < 0 ? '−' : '') : (w < 0 ? ' − ' : ' + ')) + '<b>' + Math.abs(w).toFixed(2) + '</b>·' + xs;
    });
    var b = net.b[l][n.j];
    terms += (b < 0 ? ' − ' : ' + ') + '<b>' + Math.abs(b).toFixed(2) + '</b>';
    $('nodeMath').innerHTML = (isOut ? 'ŷ' : 'a') + ' = ' + f + '( ' + terms + ' ) = ' + f + '(' + fmt(fwd.z[n.l][n.j]) +
      ') = <b>' + fmt(fwd.a[n.l][n.j]) + '</b>';

    var out = fwd.a[fwd.a.length - 1], y = data[state.point].y, L = 0;
    out.forEach(function (o, k) { L += 0.5 * (o - y[k]) * (o - y[k]); });
    $('lossMath').innerHTML = out.length === 1
      ? 'loss = ½(ŷ − y)² = ½(' + fmt(out[0]) + ' − ' + y[0] + ')² = <b>' + fmt(L, 3) + '</b>'
      : 'loss = ½Σ(ŷ − y)² = <b>' + fmt(L, 3) + '</b>';

    ['A', 'B'].forEach(function (name, k) {
      var p = state.axes[k], w = NN.getParam(net, p), gr = NN.getGrad(g, p), s = p.i < 0 ? 'b' : 'w';
      $('updMath' + name).innerHTML = '<span class="tag">' + name + '</span><span>' + s + ' − η·∂L/∂' + s + ' = ' +
        fmt(w, 3) + ' − ' + state.lr.toFixed(1) + '·(' + fmt(gr, 4) + ') = <b>' + fmt(w - state.lr * gr, 3) + '</b></span>';
    });
  }

  // ------------------------------------------------------------ data table
  var rows = [];
  function buildTable() {
    var nIn = state.sizes[0], nOut = state.sizes[state.sizes.length - 1], t = $('data'), k;
    var sub = '₁₂₃₄';
    var h = '<thead><tr>';
    for (k = 0; k < nIn; k++) h += '<th>x' + sub[k] + '</th>';
    h += '<th class="gap" colspan="' + nOut + '">target</th><th class="gap">prediction</th></tr></thead><tbody>';
    data.forEach(function (pt, r) {
      h += '<tr data-r="' + r + '">';
      pt.x.forEach(function (v) { h += '<td>' + v + '</td>'; });
      pt.y.forEach(function (v, o) {
        h += '<td' + (o === 0 ? ' class="gap"' : '') + '><button type="button" class="ybtn" data-o="' + o +
             '" title="Flip this target"></button></td>';
      });
      h += '<td class="gap">';
      for (var o = 0; o < nOut; o++) h += '<span class="pred"><span class="track"><i class="t"></i><i class="p"></i></span><span class="v"></span></span> ';
      h += '</td></tr>';
    });
    t.innerHTML = h + '</tbody>';
    rows = Array.prototype.map.call(t.querySelectorAll('tbody tr'), function (tr) {
      return { tr: tr, y: tr.querySelectorAll('.ybtn'), t: tr.querySelectorAll('.t'), p: tr.querySelectorAll('.p'), v: tr.querySelectorAll('.v') };
    });
  }

  $('data').addEventListener('click', function (e) {
    var tr = e.target.closest('tr[data-r]');
    if (!tr) return;
    var r = +tr.getAttribute('data-r');
    if (e.target.classList.contains('ybtn')) {
      var o = +e.target.getAttribute('data-o');
      data[r].y[o] = 1 - data[r].y[o];
      state.hist = [NN.loss(net, data)]; state.histStride = 1; state.steps = 0;
    }
    state.point = r;
    render();
  });

  function updateTable() {
    rows.forEach(function (row, r) {
      var out = NN.forward(net, data[r].x).a[net.W.length];
      row.tr.classList.toggle('cur', r === state.point);
      data[r].y.forEach(function (y, o) {
        row.y[o].textContent = y;
        row.t[o].style.left = (y * 100) + '%';
        row.p[o].style.left = (out[o] * 100) + '%';
        row.v[o].textContent = out[o].toFixed(2);
      });
    });
  }

  // ---------------------------------------------------------- loss surface
  var GRID = 26, RAMP = [[226, 238, 252], [134, 182, 239], [42, 120, 214], [16, 66, 129]];
  function ramp(t) {
    t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    var k = Math.min(RAMP.length - 2, Math.floor(t)), f = t - k;
    return 'rgb(' + RAMP[k].map(function (v, c) { return Math.round(v + (RAMP[k + 1][c] - v) * f); }).join(',') + ')';
  }

  function drawSurface(g) {
    var cv = $('surface'), dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var pA = state.axes[0], pB = state.axes[1], va = NN.getParam(net, pA), vb = NN.getParam(net, pB);
    state.range = Math.max(state.range, Math.ceil(Math.max(Math.abs(va), Math.abs(vb)) + 0.5));
    var Rg = state.range, pts = lossPoints(), i, j;

    // Loss at every grid position of (A, B), all other weights held where they are.
    var Z = [], zMax = 1e-6, zMin = Infinity;
    for (i = 0; i < GRID; i++) {
      Z.push([]);
      NN.setParam(net, pA, -Rg + 2 * Rg * i / (GRID - 1));
      for (j = 0; j < GRID; j++) {
        NN.setParam(net, pB, -Rg + 2 * Rg * j / (GRID - 1));
        var L = NN.loss(net, pts);
        Z[i].push(L);
        if (L > zMax) zMax = L;
        if (L < zMin) zMin = L;
      }
    }
    NN.setParam(net, pA, va);
    NN.setParam(net, pB, vb);

    var cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    var S = Math.min(w * 0.33, h * 0.38), HS = 0.75, ox = w / 2, oy = h * 0.64;
    function proj(u, v, z) {            // u, v in [-1, 1]; z in [0, 1]
      var x1 = u * cy - v * sy, y1 = u * sy + v * cy;
      return [ox + x1 * S, oy - y1 * sp * S - z * HS * cp * S, y1];
    }
    function heightAt(a, b) {           // bilinear lookup on the grid
      var fi = (Math.max(-Rg, Math.min(Rg, a)) + Rg) / (2 * Rg) * (GRID - 1);
      var fj = (Math.max(-Rg, Math.min(Rg, b)) + Rg) / (2 * Rg) * (GRID - 1);
      var i0 = Math.min(GRID - 2, Math.floor(fi)), j0 = Math.min(GRID - 2, Math.floor(fj)), di = fi - i0, dj = fj - j0;
      return (Z[i0][j0] * (1 - di) * (1 - dj) + Z[i0 + 1][j0] * di * (1 - dj) + Z[i0][j0 + 1] * (1 - di) * dj + Z[i0 + 1][j0 + 1] * di * dj) / zMax;
    }
    function onSurface(a, b, lift) {
      var q = proj(Math.max(-1, Math.min(1, a / Rg)), Math.max(-1, Math.min(1, b / Rg)), heightAt(a, b));
      return [q[0], q[1] - (lift || 0)];
    }

    var ink3 = '#8a8983', ink = '#1f1f1e';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';

    // floor outline and the vertical loss axis at the far corner
    var c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(function (q) { return proj(q[0], q[1], 0); });
    ctx.strokeStyle = '#cfcec9'; ctx.lineWidth = 1;
    ctx.beginPath(); c.forEach(function (q, k) { if (k) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); }); ctx.closePath(); ctx.stroke();
    var fu = sy >= 0 ? 1 : -1, fv = cy >= 0 ? 1 : -1, f0 = proj(fu, fv, 0), f1 = proj(fu, fv, 1);
    ctx.beginPath(); ctx.moveTo(f0[0], f0[1]); ctx.lineTo(f1[0], f1[1]); ctx.stroke();
    ctx.fillStyle = ink3;
    ctx.fillText('loss ' + zMax.toFixed(zMax < 0.1 ? 3 : 2), f1[0], f1[1] - 10);

    // surface, far quads first
    var quads = [];
    for (i = 0; i < GRID - 1; i++) for (j = 0; j < GRID - 1; j++) {
      var u0 = -1 + 2 * i / (GRID - 1), u1 = -1 + 2 * (i + 1) / (GRID - 1);
      var v0 = -1 + 2 * j / (GRID - 1), v1 = -1 + 2 * (j + 1) / (GRID - 1);
      // height is measured from zero; colour stretches over the range actually on screen
      var zm = zMax - zMin < 1e-9 ? 0.4 : ((Z[i][j] + Z[i + 1][j] + Z[i + 1][j + 1] + Z[i][j + 1]) / 4 - zMin) / (zMax - zMin);
      quads.push({
        d: ((u0 + u1) * sy + (v0 + v1) * cy) / 2, z: zm,
        p: [proj(u0, v0, Z[i][j] / zMax), proj(u1, v0, Z[i + 1][j] / zMax), proj(u1, v1, Z[i + 1][j + 1] / zMax), proj(u0, v1, Z[i][j + 1] / zMax)]
      });
    }
    quads.sort(function (a, b) { return b.d - a.d; });
    ctx.lineWidth = 0.6; ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineJoin = 'round';
    quads.forEach(function (q) {
      ctx.beginPath();
      ctx.moveTo(q.p[0][0], q.p[0][1]);
      for (var k = 1; k < 4; k++) ctx.lineTo(q.p[k][0], q.p[k][1]);
      ctx.closePath();
      ctx.fillStyle = ramp(q.z);
      ctx.fill(); ctx.stroke();
    });

    // axis names and ranges along the two near edges
    var nv = cy >= 0 ? -1 : 1, nu = sy >= 0 ? -1 : 1;
    function axisLabel(name, mid, lo, hi) {
      var m = proj(mid[0] * 1.22, mid[1] * 1.22, 0), a = proj(lo[0], lo[1], 0), b = proj(hi[0], hi[1], 0);
      ctx.fillStyle = ink3;
      ctx.fillText('−' + Rg, a[0], a[1] + 8); ctx.fillText(String(Rg), b[0], b[1] + 8);
      ctx.fillStyle = ink;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(m[0] - 8, m[1], 16, 16, 4) : ctx.rect(m[0] - 8, m[1], 16, 16); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '600 10px system-ui, sans-serif';
      ctx.fillText(name, m[0], m[1] + 8.5);
      ctx.font = '11px system-ui, sans-serif';
    }
    axisLabel('A', [0, nv], [-0.8, nv * 1.1], [0.8, nv * 1.1]);
    axisLabel('B', [nu, 0], [nu * 1.1, -0.8], [nu * 1.1, 0.8]);

    // the path walked so far
    var mark = '#eb6834';
    if (state.trail.length > 1) {
      ctx.strokeStyle = mark; ctx.lineWidth = 2; ctx.beginPath();
      state.trail.forEach(function (t, k) { var q = onSurface(t[0], t[1], 2); if (k) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
      ctx.stroke();
    }

    // downhill arrow: the direction this step will move A and B
    var ga = NN.getGrad(g, pA), gb = NN.getGrad(g, pB), gn = Math.sqrt(ga * ga + gb * gb);
    var here = onSurface(va, vb, 2);
    if (gn > 1e-7) {
      var len = 0.3 * Rg, tip = onSurface(va - ga / gn * len, vb - gb / gn * len, 2);
      var ang = Math.atan2(tip[1] - here[1], tip[0] - here[0]);
      if (Math.hypot(tip[0] - here[0], tip[1] - here[1]) > 6) {
        [['#fff', 5], [ink, 2]].forEach(function (st) {
          ctx.strokeStyle = st[0]; ctx.lineWidth = st[1]; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(here[0], here[1]); ctx.lineTo(tip[0], tip[1]);
          ctx.moveTo(tip[0] - 8 * Math.cos(ang - 0.45), tip[1] - 8 * Math.sin(ang - 0.45)); ctx.lineTo(tip[0], tip[1]);
          ctx.lineTo(tip[0] - 8 * Math.cos(ang + 0.45), tip[1] - 8 * Math.sin(ang + 0.45));
          ctx.stroke();
        });
      }
    }
    // where the network is now
    ctx.beginPath(); ctx.arc(here[0], here[1], 6, 0, Math.PI * 2);
    ctx.fillStyle = mark; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();

    // sliders
    [['axisA', va], ['axisB', vb]].forEach(function (s) {
      var inp = $(s[0]);
      inp.min = -Rg; inp.max = Rg;
      if (document.activeElement !== inp || state.running) inp.value = s[1];
      $(s[0] + 'Out').textContent = fmt(s[1]);
    });
  }

  (function rotate() {
    var cv = $('surface'), drag = null;
    cv.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', function (e) {
      if (!drag) return;
      state.yaw += (e.clientX - drag[0]) * 0.01;
      state.pitch = Math.max(0.12, Math.min(1.45, state.pitch + (e.clientY - drag[1]) * 0.01));
      drag = [e.clientX, e.clientY];
      render();
    });
    ['pointerup', 'pointercancel'].forEach(function (n) { cv.addEventListener(n, function () { drag = null; }); });
  })();

  ['A', 'B'].forEach(function (name, k) {
    $('axis' + name).addEventListener('input', function (e) {
      NN.setParam(net, state.axes[k], +e.target.value);
      state.trail.push([NN.getParam(net, state.axes[0]), NN.getParam(net, state.axes[1])]);
      render();
    });
  });

  // ------------------------------------------------------------ loss curve
  var hover = null;
  function drawCurve() {
    var hst = state.hist, max = Math.max.apply(null, hst.concat([1e-6])), n = hst.length;
    var pts = hst.map(function (v, k) { return (n === 1 ? 0 : 300 * k / (n - 1)).toFixed(1) + ',' + (118 - 98 * v / max).toFixed(1); });
    if (n === 1) pts.push('300,' + pts[0].split(',')[1]);
    $('curveLine').setAttribute('points', pts.join(' '));
    $('lossMax').textContent = max.toFixed(3);
    var cur = $('curveCursor');
    if (hover !== null && n > 1) {
      var k = Math.round(hover * (n - 1));
      cur.setAttribute('x1', 300 * k / (n - 1)); cur.setAttribute('x2', 300 * k / (n - 1));
      cur.setAttribute('visibility', 'visible');
      $('lossNow').textContent = 'step ' + (k * state.histStride) + ' · ' + hst[k].toFixed(4);
    } else {
      cur.setAttribute('visibility', 'hidden');
      $('lossNow').textContent = 'step ' + state.steps + ' · ' + NN.loss(net, data).toFixed(4);
    }
  }
  $('curve').addEventListener('pointermove', function (e) {
    var r = e.currentTarget.getBoundingClientRect();
    hover = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    drawCurve();
  });
  $('curve').addEventListener('pointerleave', function () { hover = null; drawCurve(); });

  // ---------------------------------------------------------------- render
  function render() {
    var fwd = NN.forward(net, data[state.point].x);
    var g = NN.gradient(net, lossPoints());
    updateGraphs(fwd, g);
    updateMath(fwd, g);
    updateTable();
    drawSurface(g);
    drawCurve();
    var hidden = state.sizes.length - 2;
    $('layerCount').textContent = hidden;
    $('layerMinus').disabled = hidden === 0;
    $('layerPlus').disabled = hidden === MAX.layers;
  }

  // -------------------------------------------------------------- controls
  var last = 0, acc = 0;
  function tick(t) {
    if (!state.running) return;
    acc += Math.min(0.1, (t - last) / 1000) * SPEEDS[+$('speed').value];
    last = t;
    var n = Math.floor(acc);
    acc -= n;
    for (var k = 0; k < n; k++) trainStep();
    if (n) render();
    requestAnimationFrame(tick);
  }
  function setRunning(on) {
    state.running = on;
    $('run').textContent = on ? 'Pause' : 'Run';
    if (on) { last = performance.now(); acc = 1; requestAnimationFrame(tick); }
  }

  $('step').addEventListener('click', function () { setRunning(false); trainStep(); render(); });
  $('run').addEventListener('click', function () { setRunning(!state.running); });
  $('reset').addEventListener('click', function () { setRunning(false); state.seed++; rebuild(false); });

  $('lr').addEventListener('input', function (e) { state.lr = +e.target.value; $('lrOut').textContent = state.lr.toFixed(1); render(); });
  $('act').addEventListener('change', function (e) { state.act = e.target.value; rebuild(false); });
  NN.TASK_NAMES.forEach(function (n) { var o = document.createElement('option'); o.textContent = n; $('task').appendChild(o); });
  $('task').addEventListener('change', function (e) { state.task = e.target.value; rebuild(true); });

  Array.prototype.forEach.call(document.querySelectorAll('.seg button'), function (b) {
    b.addEventListener('click', function () {
      state.mode = b.getAttribute('data-mode');
      Array.prototype.forEach.call(document.querySelectorAll('.seg button'), function (o) { o.classList.toggle('is-on', o === b); });
      state.trail = [];
      render();
    });
  });

  $('layerPlus').addEventListener('click', function () {
    state.sizes.splice(state.sizes.length - 1, 0, 3);
    rebuild(false);
  });
  $('layerMinus').addEventListener('click', function () {
    state.sizes.splice(state.sizes.length - 2, 1);
    rebuild(false);
  });

  window.addEventListener('resize', function () { render(); });
  rebuild(true);
})();
