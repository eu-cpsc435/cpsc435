/**
 * qlearning.js -- the algorithm.
 *
 * The whole method is one line, applied once per step:
 *
 *     Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
 *                        \____________________  ____________________/
 *                                             \/
 *                                      the TD error: how wrong
 *                                      our old estimate was
 *
 * Everything else in this file is bookkeeping so the UI can replay training
 * one step at a time.
 */


/** Small seeded PRNG so a demo run is reproducible in class. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Values used when a hyperparameter is switched on. The UI can switch gamma,
 * epsilon and the step penalty off, in which case it passes the neutral values
 * instead (gamma 1, epsilon 0, penalty 0) so the term has no effect.
 */
const DEFAULT_PARAMS = {
  alpha: 0.2,        // learning rate
  gamma: 0.9,        // discount factor
  epsilonStart: 1.0, // exploration at episode 0
  epsilonDecay: 0.95,
  epsilonMin: 0.05,
  stepPenalty: -0.04,
  maxSteps: 60,
  seed: 20250921,
};

function epsilonFor(params, episodeIndex) {
  const eps = params.epsilonStart * Math.pow(params.epsilonDecay, episodeIndex);
  return Math.max(params.epsilonMin, eps);
}

/** Index into the flat Q array. */
const qi = (s, a) => s * NUM_ACTIONS + a;

/**
 * Both of these range over `actions` -- the moves that are actually available
 * from that cell -- not over all four directions. An action that would walk into
 * a wall or off the edge is not part of the decision or of max Q(s',.).
 */
function rowMax(q, s, actions) {
  let best = -Infinity;
  for (const a of actions) best = Math.max(best, q[qi(s, a)]);
  return best === -Infinity ? 0 : best;   // terminal cell: nothing follows it
}

function greedyAction(q, s, actions) {
  if (!actions.length) return -1;
  let best = actions[0];
  for (const a of actions) if (q[qi(s, a)] > q[qi(s, best)]) best = a;
  return best;
}

/**
 * Owns the environment, the Q-table, and the full recorded history of training
 * so the UI can scrub backwards and forwards through it.
 */
class Trainer {
  constructor(world, params) {
    this.world = world;
    this.params = { ...params };
    this.q = new Float64Array(world.numStates * NUM_ACTIONS);
    this.updates = new Int32Array(world.numStates * NUM_ACTIONS);
    this.steps = [];     // flat list of every update ever made
    this.episodes = [];  // { start, count, totalReward, epsilon, outcome }
    this.rng = mulberry32(this.params.seed);
    this._cache = null;
  }

  /** Swap in a new world but keep everything learned -- the "does it generalize?" demo. */
  setWorld(world) {
    this.world = world;
  }

  resetQ() {
    this.q.fill(0);
    this.updates.fill(0);
    this.steps = [];
    this.episodes = [];
    this.rng = mulberry32(this.params.seed);
    this._cache = null;
  }

  get totalSteps() { return this.steps.length; }

  chooseAction(s, epsilon) {
    const actions = this.world.validActions(s);
    if (this.rng() < epsilon) {
      return {
        action: actions[Math.floor(this.rng() * actions.length)],
        greedy: false,
        ties: 0,
      };
    }
    // Break ties randomly so an all-zero row does not always pick the first
    // direction. With exploration switched off this is the only source of
    // variety early on, so the number of tied actions is worth reporting.
    const best = this.q[qi(s, greedyAction(this.q, s, actions))];
    const ties = actions.filter((a) => this.q[qi(s, a)] === best);
    return {
      action: ties[Math.floor(this.rng() * ties.length)],
      greedy: true,
      ties: ties.length,
    };
  }

  /** One application of the Q-learning update rule. Returns a record of it. */
  applyUpdate(s, a, reward, next, done, epsilon, greedy, ties) {
    const { alpha, gamma } = this.params;
    const qOld = this.q[qi(s, a)];
    const maxNext = done ? 0 : rowMax(this.q, next, this.world.validActions(next));
    const target = reward + gamma * maxNext;
    const tdError = target - qOld;
    const qNew = qOld + alpha * tdError;

    this.q[qi(s, a)] = qNew;
    this.updates[qi(s, a)] += 1;

    return { s, a, reward, next, done, qOld, maxNext, target, tdError, qNew, epsilon, greedy, ties };
  }

  runEpisode() {
    const episodeIndex = this.episodes.length;
    const epsilon = epsilonFor(this.params, episodeIndex);
    const start = this.steps.length;

    let s = this.world.start;
    let totalReward = 0;
    let outcome = 'timeout';

    for (let t = 0; t < this.params.maxSteps; t++) {
      if (!this.world.validActions(s).length) { outcome = 'stuck'; break; }
      const { action, greedy, ties } = this.chooseAction(s, epsilon);
      const { next, reward, done } = this.world.step(s, action);
      this.steps.push(this.applyUpdate(s, action, reward, next, done, epsilon, greedy, ties));
      totalReward += reward;
      s = next;
      if (done) { outcome = reward > 0 ? 'goal' : 'trap'; break; }
    }

    const ep = { start, count: this.steps.length - start, totalReward, epsilon, outcome };
    this.episodes.push(ep);
    return ep;
  }

  runEpisodes(n) {
    for (let i = 0; i < n; i++) this.runEpisode();
  }

  /**
   * The Q-table as it stood after the first `n` updates. Used for scrubbing.
   * Cached and extended incrementally, since we usually step forwards.
   */
  snapshotAt(n) {
    n = Math.max(0, Math.min(n, this.steps.length));
    let cache = this._cache;
    if (!cache || cache.n > n) {
      cache = {
        n: 0,
        q: new Float64Array(this.q.length),
        updates: new Int32Array(this.updates.length),
      };
    }
    for (let i = cache.n; i < n; i++) {
      const st = this.steps[i];
      cache.q[qi(st.s, st.a)] = st.qNew;
      cache.updates[qi(st.s, st.a)] += 1;
    }
    cache.n = n;
    this._cache = cache;
    return cache;
  }

  /** Follow the greedy policy of a given Q-table from the start state. */
  greedyRollout(q, maxSteps = 60) {
    const path = [];
    const seen = new Set();
    let s = this.world.start;
    let outcome = 'timeout';

    for (let t = 0; t < maxSteps; t++) {
      const key = s;
      if (seen.has(key)) { outcome = 'loop'; break; }
      seen.add(key);

      const a = greedyAction(q, s, this.world.validActions(s));
      if (a < 0) { outcome = 'stuck'; break; }
      const { next, reward, done } = this.world.step(s, a);
      path.push({ s, a, next, reward, done, qValue: q[qi(s, a)] });
      s = next;
      if (done) { outcome = reward > 0 ? 'goal' : 'trap'; break; }
    }
    return { path, outcome };
  }
}
