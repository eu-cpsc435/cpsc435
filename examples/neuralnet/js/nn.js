/* A tiny fully-connected neural network. No DOM, no dependencies.
 *
 *   net.sizes  [2, 3, 1]            neurons per layer, inputs first
 *   net.W[l][j][i]                  weight from neuron i of layer l into neuron j of layer l+1
 *   net.b[l][j]                     bias of neuron j of layer l+1
 *
 * Hidden layers use net.act; the output layer is always a sigmoid.
 * Loss for one data point is  L = 1/2 * sum_k (yhat_k - y_k)^2.
 */
var NN = (function () {
  'use strict';

  // Seeded random numbers, so a reset reproduces the same starting weights.
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Each activation: f(z), and its derivative written in terms of a = f(z).
  var ACT = {
    sigmoid: { label: 'σ',   f: function (z) { return 1 / (1 + Math.exp(-z)); }, d: function (a) { return a * (1 - a); } },
    tanh:    { label: 'tanh', f: function (z) { return Math.tanh(z); },           d: function (a) { return 1 - a * a; } },
    relu:    { label: 'relu', f: function (z) { return z > 0 ? z : 0; },          d: function (a) { return a > 0 ? 1 : 0; } }
  };

  function create(sizes, act, seed) {
    var rand = rng(seed || 1);
    var net = { sizes: sizes.slice(), act: act || 'sigmoid', W: [], b: [] };
    for (var l = 0; l < sizes.length - 1; l++) {
      var Wl = [], bl = [];
      for (var j = 0; j < sizes[l + 1]; j++) {
        var row = [];
        for (var i = 0; i < sizes[l]; i++) row.push(rand() * 2 - 1);
        Wl.push(row);
        bl.push(rand() * 2 - 1);
      }
      net.W.push(Wl);
      net.b.push(bl);
    }
    return net;
  }

  function actOf(net, l) {            // activation used by layer l+1
    return l === net.W.length - 1 ? ACT.sigmoid : ACT[net.act];
  }

  // Forward pass. Returns every layer's weighted sums z and activations a.
  function forward(net, x) {
    var a = [x.slice()], z = [null];
    for (var l = 0; l < net.W.length; l++) {
      var f = actOf(net, l).f, prev = a[l], zl = [], al = [];
      for (var j = 0; j < net.W[l].length; j++) {
        var sum = net.b[l][j];                                   // the linear combination:
        for (var i = 0; i < prev.length; i++) sum += net.W[l][j][i] * prev[i];   // z = b + sum w_i * a_i
        zl.push(sum);
        al.push(f(sum));
      }
      z.push(zl);
      a.push(al);
    }
    return { a: a, z: z };
  }

  function pointLoss(out, y) {
    var L = 0;
    for (var k = 0; k < y.length; k++) L += 0.5 * (out[k] - y[k]) * (out[k] - y[k]);
    return L;
  }

  // Mean loss over a list of points [{x, y}].
  function loss(net, points) {
    var L = 0;
    for (var p = 0; p < points.length; p++) {
      var f = forward(net, points[p].x);
      L += pointLoss(f.a[f.a.length - 1], points[p].y);
    }
    return L / points.length;
  }

  // Backpropagation, averaged over a list of points.
  // delta[l][j] is dL/dz for neuron j of layer l+1; gW and gb mirror W and b.
  function gradient(net, points) {
    var nL = net.W.length, l, j, i, p;
    var gW = [], gb = [], delta = [];
    for (l = 0; l < nL; l++) {
      gW.push(net.W[l].map(function (row) { return row.map(function () { return 0; }); }));
      gb.push(net.b[l].map(function () { return 0; }));
      delta.push(net.b[l].map(function () { return 0; }));
    }
    var L = 0, n = points.length;

    for (p = 0; p < n; p++) {
      var f = forward(net, points[p].x), y = points[p].y;
      var out = f.a[nL];
      L += pointLoss(out, y);

      var d = new Array(nL);
      // Output layer: dL/dz = (yhat - y) * sigmoid'(z)
      d[nL - 1] = out.map(function (o, k) { return (o - y[k]) * ACT.sigmoid.d(o); });
      // Hidden layers: pass the error back through the weights.
      for (l = nL - 2; l >= 0; l--) {
        var dAct = actOf(net, l).d;
        d[l] = [];
        for (j = 0; j < net.W[l].length; j++) {
          var s = 0;
          for (var k2 = 0; k2 < net.W[l + 1].length; k2++) s += net.W[l + 1][k2][j] * d[l + 1][k2];
          d[l].push(s * dAct(f.a[l + 1][j]));
        }
      }
      // dL/dw = delta of the neuron it feeds * activation it carries; dL/db = delta.
      for (l = 0; l < nL; l++) {
        for (j = 0; j < net.W[l].length; j++) {
          for (i = 0; i < net.W[l][j].length; i++) gW[l][j][i] += d[l][j] * f.a[l][i] / n;
          gb[l][j] += d[l][j] / n;
          delta[l][j] += d[l][j] / n;
        }
      }
    }
    return { gW: gW, gb: gb, delta: delta, loss: L / n };
  }

  // One step of gradient descent:  w <- w - lr * dL/dw
  function step(net, g, lr) {
    for (var l = 0; l < net.W.length; l++) {
      for (var j = 0; j < net.W[l].length; j++) {
        for (var i = 0; i < net.W[l][j].length; i++) net.W[l][j][i] -= lr * g.gW[l][j][i];
        net.b[l][j] -= lr * g.gb[l][j];
      }
    }
  }

  // A parameter is addressed as {l, j, i}; i === -1 means the bias.
  function getParam(net, p) { return p.i < 0 ? net.b[p.l][p.j] : net.W[p.l][p.j][p.i]; }
  function setParam(net, p, v) { if (p.i < 0) net.b[p.l][p.j] = v; else net.W[p.l][p.j][p.i] = v; }
  function getGrad(g, p) { return p.i < 0 ? g.gb[p.l][p.j] : g.gW[p.l][p.j][p.i]; }

  // Truth-table tasks that work for any number of inputs.
  var TASKS = {
    AND: function (x) { return x.every(function (v) { return v === 1; }) ? 1 : 0; },
    OR:  function (x) { return x.some(function (v) { return v === 1; }) ? 1 : 0; },
    XOR: function (x) { return x.reduce(function (s, v) { return s + v; }, 0) % 2; }
  };
  var TASK_NAMES = ['AND', 'OR', 'XOR'];

  // Every 0/1 combination of the inputs. Extra outputs take the next tasks in the list.
  function makeData(nIn, nOut, task) {
    var start = TASK_NAMES.indexOf(task), pts = [];
    for (var n = 0; n < (1 << nIn); n++) {
      var x = [];
      for (var i = nIn - 1; i >= 0; i--) x.push((n >> i) & 1);
      var y = [];
      for (var k = 0; k < nOut; k++) y.push(TASKS[TASK_NAMES[(start + k) % TASK_NAMES.length]](x));
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  return {
    ACT: ACT, TASK_NAMES: TASK_NAMES,
    create: create, forward: forward, loss: loss, gradient: gradient, step: step,
    getParam: getParam, setParam: setParam, getGrad: getGrad, makeData: makeData
  };
})();

if (typeof module !== 'undefined') module.exports = NN;
