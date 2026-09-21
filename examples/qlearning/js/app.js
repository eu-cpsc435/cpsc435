/**
 * app.js -- all of the UI. The interesting part of the demo lives in
 * qlearning.js; this file just draws it.
 */


const $ = (id) => document.getElementById(id);

/* --------------------------------------------------------------- state */

// Slider values, used only when the matching switch below is on.
const params = { ...DEFAULT_PARAMS };

// Start with the extra terms switched off, so the first thing on screen is the
// plainest form of the rule: Q <- Q + alpha [ r + max Q(s',.) - Q ].
const enabled = { gamma: false, epsilon: false, stepPenalty: false };

/** What the algorithm actually runs with: a switched-off term gets its neutral value. */
function effectiveParams() {
  return {
    ...params,
    gamma: enabled.gamma ? params.gamma : 1,
    epsilonStart: enabled.epsilon ? params.epsilonStart : 0,
    epsilonDecay: enabled.epsilon ? params.epsilonDecay : 1,
    epsilonMin: enabled.epsilon ? params.epsilonMin : 0,
    stepPenalty: enabled.stepPenalty ? params.stepPenalty : 0,
  };
}

let layout = LAYOUTS[0];
let world = new GridWorld(layout, { stepPenalty: effectiveParams().stepPenalty });
let trainer = new Trainer(world, effectiveParams());

let mode = 'training';
let pos = { ep: 0, step: -1 };          // scrub position in recorded training
let exec = { path: [], outcome: null, idx: -1 };

/* --------------------------------------------------------------- utils */

const fmt2 = (n) => (Object.is(n, -0) ? 0 : n).toFixed(2);
const fmt3 = (n) => (Object.is(n, -0) ? 0 : n).toFixed(3);
const label = (s) => `(${world.rowOf(s)}, ${world.colOf(s)})`;

function hasHistory() { return trainer.episodes.length > 0; }

function currentEpisode() {
  return hasHistory() ? trainer.episodes[pos.ep] : null;
}

/** How many updates have been applied at the current scrub position. */
function globalIndex() {
  const ep = currentEpisode();
  if (!ep) return 0;
  return ep.start + pos.step + 1;
}

function selectedStep() {
  const ep = currentEpisode();
  if (!ep || pos.step < 0) return null;
  return trainer.steps[ep.start + pos.step];
}

/** The step being highlighted on screen. Execution mode applies no updates. */
function highlightedStep() {
  return mode === 'training' ? selectedStep() : null;
}

/** Q-table and visit counts as of the current scrub position. */
function snapshot() {
  return trainer.snapshotAt(globalIndex());
}

function agentState() {
  if (mode === 'execution') {
    return exec.idx < 0 ? world.start : exec.path[exec.idx].next;
  }
  const step = selectedStep();
  return step ? step.next : world.start;
}

/* ------------------------------------------------------------ svg world */

const CELL = 84;
const GAP = 22;
const PAD = 4;
const SVG_NS = 'http://www.w3.org/2000/svg';

const cellX = (c) => PAD + c * (CELL + GAP);
const cellY = (r) => PAD + r * (CELL + GAP);

function el(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function cellFill(s) {
  if (world.isWall(s)) return 'var(--cell-wall)';
  if (world.isGoal(s)) return 'var(--cell-goal)';
  if (world.isTrap(s)) return 'var(--cell-trap)';
  return 'var(--cell)';
}

function renderWorld() {
  const svg = $('world');
  const w = world.cols * CELL + (world.cols - 1) * GAP + PAD * 2;
  const h = world.rows * CELL + (world.rows - 1) * GAP + PAD * 2;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.replaceChildren();

  const { q, updates } = snapshot();
  const showBars = $('showBars').checked;
  const showPolicy = $('showPolicy').checked || mode === 'execution';
  const active = highlightedStep();

  // scale bar thickness against the largest magnitude currently in the table
  let maxAbs = 0.05;
  for (let i = 0; i < q.length; i++) {
    if (updates[i] > 0) maxAbs = Math.max(maxAbs, Math.abs(q[i]));
  }

  // --- cells
  for (let s = 0; s < world.numStates; s++) {
    const x = cellX(world.colOf(s));
    const y = cellY(world.rowOf(s));
    svg.appendChild(el('rect', { x, y, width: CELL, height: CELL, fill: cellFill(s) }));

    if (world.isTerminal(s)) {
      const t = el('text', {
        x: x + CELL / 2, y: y + CELL / 2 + 7,
        'text-anchor': 'middle',
        'font-family': 'ui-monospace, monospace',
        'font-size': 22, 'font-weight': 600,
        fill: 'rgba(255,255,255,0.92)',
      });
      t.textContent = world.isGoal(s) ? '+1' : '−1';
      svg.appendChild(t);
    }
  }

  // --- Q-value bars, one per edge
  if (showBars) {
    const inset = 5;
    const edge = 3;
    for (let s = 0; s < world.numStates; s++) {
      if (world.isWall(s) || world.isTerminal(s)) continue;
      const x = cellX(world.colOf(s));
      const y = cellY(world.rowOf(s));

      for (const a of world.validActions(s)) {
        const i = qi(s, a);
        if (updates[i] === 0) continue;

        const v = q[i];
        const ratio = Math.min(1, Math.abs(v) / maxAbs);
        const thick = 4 + 8 * ratio;
        const len = CELL - inset * 2;
        const color = v < -1e-9 ? 'var(--q-neg)' : 'var(--q-pos)';

        let box;
        if (a === 0) box = { x: x + inset, y: y + edge, width: len, height: thick };
        else if (a === 1) box = { x: x + CELL - edge - thick, y: y + inset, width: thick, height: len };
        else if (a === 2) box = { x: x + inset, y: y + CELL - edge - thick, width: len, height: thick };
        else box = { x: x + edge, y: y + inset, width: thick, height: len };

        const bar = el('rect', {
          ...box, fill: color, rx: 1,
          opacity: (0.45 + 0.55 * ratio).toFixed(2),
        });
        if (active && active.s === s && active.a === a) {
          bar.setAttribute('opacity', '1');
          bar.setAttribute('stroke', '#1f2933');
          bar.setAttribute('stroke-width', '1.5');
        }
        svg.appendChild(bar);
      }
    }
  }

  // --- greedy policy arrows
  if (showPolicy) {
    for (let s = 0; s < world.numStates; s++) {
      if (world.isWall(s) || world.isTerminal(s)) continue;
      const valid = world.validActions(s);
      if (!valid.some((a) => updates[qi(s, a)] > 0)) continue;

      const a = greedyAction(q, s, valid);
      const cx = cellX(world.colOf(s)) + CELL / 2;
      const cy = cellY(world.rowOf(s)) + CELL / 2;
      const arrow = el('path', {
        d: 'M 0 -13 L 9 5 L 0 0 L -9 5 Z',
        fill: 'rgba(31,41,51,0.62)',
        transform: `translate(${cx} ${cy}) rotate(${a * 90})`,
      });
      svg.appendChild(arrow);
    }
  }

  // --- highlight the state being updated
  if (active) {
    svg.appendChild(el('rect', {
      x: cellX(world.colOf(active.s)) - 3,
      y: cellY(world.rowOf(active.s)) - 3,
      width: CELL + 6, height: CELL + 6,
      fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, rx: 3,
    }));
  }

  // --- agent
  const s = agentState();
  svg.appendChild(el('circle', {
    id: 'agent',
    cx: cellX(world.colOf(s)) + CELL / 2,
    cy: cellY(world.rowOf(s)) + CELL / 2,
    r: CELL * 0.32,
    fill: 'var(--agent)',
    stroke: 'rgba(0,0,0,0.12)', 'stroke-width': 1,
  }));
}

/* ----------------------------------------------------------- q-table ui */

function renderQTable() {
  const body = $('qtableBody');
  const { q, updates } = snapshot();
  const active = highlightedStep();
  const here = agentState();
  const rows = [];

  for (const s of world.activeStates()) {
    const cls = [];
    if (s === here) cls.push('current');
    if (world.isTerminal(s)) cls.push('terminal');

    let badge = '';
    if (world.isGoal(s)) badge = '<span class="badge badge-goal">+1</span>';
    if (world.isTrap(s)) badge = '<span class="badge badge-trap">&minus;1</span>';

    const valid = world.validActions(s);
    const best = greedyAction(q, s, valid);
    const anyVisit = valid.some((a) => updates[qi(s, a)] > 0);

    const cells = ACTIONS.map((_, a) => {
      // An action that cannot be taken from this cell has no Q-value at all.
      if (!valid.includes(a)) return '<td class="na" title="not a legal move here">&mdash;</td>';
      const tdCls = [];
      if (updates[qi(s, a)] === 0) tdCls.push('zero');
      if (anyVisit && a === best) tdCls.push('best');
      if (active && active.s === s && active.a === a) tdCls.push('updated');
      return `<td class="${tdCls.join(' ')}">${fmt2(q[qi(s, a)])}</td>`;
    }).join('');

    rows.push(
      `<tr class="${cls.join(' ')}"><td class="state">${label(s)}${badge}</td>${cells}</tr>`
    );
  }
  body.innerHTML = rows.join('');
}

/* ----------------------------------------------------------- formula ui */

const TIPS = {
  qNew: 'The updated estimate, and the only number this step changes. Everything else on the line is read, not written.',
  qOld: 'What the table currently says about taking action a in state s. The update nudges this number, it does not replace it.',
  alpha: 'Learning rate. How far to move the estimate toward the target: 0 never learns, 1 throws the old estimate away entirely.',
  reward: 'The reward collected on entering the next state: +1 at the goal, -1 at a trap, otherwise the step penalty.',
  gamma: 'Discount factor. How much a reward one step later is worth compared to the same reward right now. Below 1 it makes the agent prefer shorter paths.',
  maxNext: 'The best value available from the next state, i.e. what the agent expects to collect from there onward. Exactly 0 when the next state is terminal, because nothing follows it.',
  assign: 'Assignment. The value on the left is replaced by the result of the right-hand side.',
  bracket: 'The TD error: the target minus the current estimate. How wrong the table was about this state-action pair.',
};

const escapeAttr = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;');

/** Render the equation as columns: each symbol sits above its own value. */
function renderEquation(cells) {
  const grid = $('eqGrid');
  grid.className = 'eq-grid';
  grid.innerHTML = cells.map((c) => {
    const cls = ['eq-cell'];
    if (c.op) cls.push('is-op');
    if (c.br) cls.push('is-op', 'is-br');
    const tip = c.tip ? ` data-tip="${escapeAttr(c.tip)}" tabindex="0"` : '';
    const val = c.val === undefined
      ? ''
      : `<span class="eq-val ${c.tone || ''}">${c.val}</span>`;
    return `<span class="${cls.join(' ')}"${tip}><span class="eq-sym">${c.sym}</span>${val}</span>`;
  }).join('');
}

function showIdleEquation(message) {
  const grid = $('eqGrid');
  grid.className = 'eq-idle';
  grid.textContent = message;
}

const tone = (v) => (v > 1e-9 ? 'pos' : v < -1e-9 ? 'neg' : '');

const ARM_CLASS = ['up', 'right', 'down', 'left'];

/**
 * Draw the four directions in the positions they point, showing the value the
 * agent read for each one, which ones were tied for best, and which it took.
 * When the taken arm is filled but a different arm is outlined, that is an
 * exploratory move: the agent knowingly ignored the better value.
 */
function renderChoice(s, values, taken, note) {
  const best = values.size ? Math.max(...values.values()) : 0;

  const arms = ACTIONS.map((act, a) => {
    const cls = ['arm', ARM_CLASS[a]];
    const has = values.has(a);
    if (!has) cls.push('is-na');
    else {
      if (values.get(a) === best) cls.push('is-best');
      if (a === taken) cls.push('is-taken');
    }
    const v = has ? fmt2(values.get(a)) : '—';
    return `<div class="${cls.join(' ')}"><span class="arm-dir">${act.symbol}</span>` +
           `<span class="arm-q">${v}</span></div>`;
  }).join('');

  $('choice').innerHTML =
    `<div class="choice-head">Actions at ${label(s)}</div>` +
    `<div class="compass">${arms}<div class="compass-center">s</div></div>` +
    `<p class="choice-note">${note}</p>`;
}

/**
 * The move the agent is about to make from where it currently stands. The
 * snapshot at this scrub position is exactly the table that decision was made
 * with, since its own update has not been applied yet.
 */
function upcomingStep() {
  if (mode === 'execution') {
    const idx = exec.idx + 1;
    return idx < exec.path.length ? exec.path[idx] : null;
  }
  const ep = currentEpisode();
  if (!ep) return null;
  const idx = pos.step + 1;
  return idx < ep.count ? trainer.steps[ep.start + idx] : null;
}

function renderChoicePanel() {
  const { q } = snapshot();
  const upcoming = upcomingStep();
  const s = upcoming ? upcoming.s : agentState();
  const values = new Map(world.validActions(s).map((a) => [a, q[qi(s, a)]]));

  let taken = -1;
  let note;

  if (world.isTerminal(s)) {
    note = 'A terminal cell has no actions &mdash; the episode ends on arrival.';
  } else if (!upcoming) {
    note = hasHistory()
      ? 'This episode ended here. Move to another episode, or train more.'
      : 'Train an episode to watch the agent choose.';
  } else {
    taken = upcoming.a;
    const best = Math.max(...values.values());
    const pick = `<b>${ACTIONS[taken].symbol} ${ACTIONS[taken].name}</b>`;

    if (mode === 'execution') {
      note = `<b>Greedy</b>: takes ${pick}, the highest value at ${fmt2(best)}.
        No exploration in this mode.`;
    } else if (!upcoming.greedy) {
      // A random pick can land on the best action by chance -- do not claim a
      // disagreement that is not on screen.
      const luckily = values.get(taken) === best;
      note = `<b>Exploring</b> (&epsilon; = ${fmt2(upcoming.epsilon)}): takes ${pick} at
        random, ignoring the values. ` + (luckily
          ? 'It happens to be a best one anyway.'
          : 'The outlined arm is what greedy would have picked.');
    } else if (upcoming.ties > 1) {
      note = `<b>Greedy</b>, but ${upcoming.ties} actions tie at ${fmt2(best)}, so ${pick}
        is picked at random from among them.`;
    } else {
      note = `<b>Greedy</b>: takes ${pick}, the highest value at ${fmt2(best)}.`;
    }
  }

  renderChoice(s, values, taken, note);
}

function term(key, value, t = '') {
  return `<span class="term ${t}"><span class="k">${key}</span><span class="v">${value}</span></span>`;
}

function renderFormula() {
  if (mode === 'execution') { renderExecutionFormula(); return; }

  const tag = $('formulaTag');
  $('formulaTitle').textContent = 'Result';
  const step = selectedStep();

  if (!step) {
    tag.textContent = 'nothing yet';
    tag.classList.remove('is-live');
    showIdleEquation(hasHistory()
      ? 'No move made yet this episode. Step forward to apply the first update.'
      : 'Train an episode, then step through it.');
    $('terms').innerHTML = '';
    $('explain').innerHTML = 'Every step of every episode applies this rule exactly once, to a single cell of the Q-table.';
    return;
  }

  const { alpha, gamma } = trainer.params;
  tag.textContent = `episode ${pos.ep + 1} · step ${pos.step + 1}`;
  tag.classList.add('is-live');

  renderEquation([
    { sym: 'Q(s,a)', val: fmt3(step.qNew), tone: 'is-result', tip: TIPS.qNew },
    { sym: '←', op: true, tip: TIPS.assign },
    { sym: 'Q(s,a)', val: fmt3(step.qOld), tip: TIPS.qOld },
    { sym: '+', op: true },
    { sym: 'α', val: fmt2(alpha), tip: TIPS.alpha },
    { sym: '[', br: true, tip: TIPS.bracket },
    { sym: 'r', val: fmt3(step.reward), tone: tone(step.reward), tip: TIPS.reward },
    { sym: '+', op: true },
    { sym: 'γ', val: fmt2(gamma), tip: TIPS.gamma },
    { sym: '·', op: true },
    { sym: 'max<sub>a′</sub> Q(s′,a′)', val: fmt3(step.maxNext), tone: tone(step.maxNext), tip: TIPS.maxNext },
    { sym: '−', op: true },
    { sym: 'Q(s,a)', val: fmt3(step.qOld), tip: TIPS.qOld },
    { sym: ']', br: true, tip: TIPS.bracket },
  ]);

  $('terms').innerHTML = [
    term('s', label(step.s)),
    term('a', `${ACTIONS[step.a].symbol} ${ACTIONS[step.a].name}`),
    term('s′', step.done ? `${label(step.next)} terminal` : label(step.next)),
    term('TD target', fmt3(step.target), tone(step.target)),
    term('TD error', (step.tdError >= 0 ? '+' : '') + fmt3(step.tdError), tone(step.tdError)),
  ].join('');

  let story;
  if (step.done) {
    story = `The agent entered a terminal cell, so there is no next state and
      <b>max Q(s′,·) = 0</b>. The target collapses to the raw reward,
      <b>${fmt3(step.reward)}</b> &mdash; this is where value first enters the table.`;
  } else if (Math.abs(step.maxNext) < 1e-9 && Math.abs(step.reward) < 1e-9) {
    story = `Reward is 0 and the next state has no learned value yet, so the target is 0
      and nothing moves. Early episodes look like this until the agent stumbles into a
      terminal cell and value starts propagating backwards.`;
  } else {
    story = `No reward here, so the whole target comes from the next state:
      <b>${fmt2(gamma)} &times; ${fmt3(step.maxNext)} = ${fmt3(step.target)}</b>.
      This is how value flows one cell backwards per visit.`;
  }

  $('explain').innerHTML = `${story} The new estimate moves <b>${fmt2(alpha)}</b> of the
    way from ${fmt3(step.qOld)} toward the target ${fmt3(step.target)}, landing on
    <b>${fmt3(step.qNew)}</b>. Hover any term above for what it means.`;
}

function renderExecutionFormula() {
  const tag = $('formulaTag');
  $('formulaTitle').textContent = 'Result';
  tag.textContent = 'learning off';
  tag.classList.remove('is-live');

  const prev = exec.idx >= 0 ? exec.path[exec.idx] : null;

  if (!prev) {
    showIdleEquation('No move made yet. Step forward to follow the greedy policy.');
    $('terms').innerHTML = '';
    $('explain').innerHTML = `Execution mode replays what was learned without changing it.
      The compass beside the grid shows the move about to be made; this panel reports the
      one just made.`;
    return;
  }

  showIdleEquation('Execution applies no update — the Q-table is frozen while the agent acts.');

  $('terms').innerHTML = [
    term('s', label(prev.s)),
    term('a', `${ACTIONS[prev.a].symbol} ${ACTIONS[prev.a].name}`),
    term('Q(s,a)', fmt3(prev.qValue), tone(prev.qValue)),
    term('r', fmt3(prev.reward), tone(prev.reward)),
    term('s′', prev.done ? `${label(prev.next)} terminal` : label(prev.next)),
  ].join('');

  const ended = prev.reward > 0 ? 'the <b>goal</b>' : 'a <b>trap</b>';
  $('explain').innerHTML = prev.done
    ? `The agent entered ${ended} and the run ended. Scrub the training sliders and come
       back here to see how the policy looked earlier in learning.`
    : `Moved to ${label(prev.next)} for a reward of ${fmt3(prev.reward)}. In training this
       step would have updated Q(s,a); here the table is read-only.`;
}

/* --------------------------------------------------------- controls ui */

function episodeSteps(ep) {
  return trainer.steps.slice(ep.start, ep.start + ep.count);
}

/**
 * Say *why* an episode ended the way it did. A trap is almost always reached by a
 * forced exploration move rather than by choice -- worth making explicit, since
 * "hit trap" on its own reads as though nothing was learned.
 */
function outcomeText(ep) {
  if (ep.outcome === 'goal') return 'reached goal';
  if (ep.outcome === 'timeout') return 'out of steps';
  if (ep.outcome === 'stuck' || ep.count === 0) return 'no legal moves';
  const last = trainer.steps[ep.start + ep.count - 1];
  return last.greedy ? 'trap, chose it' : 'trap, exploring';
}

function renderStats() {
  $('statEpisodes').textContent = trainer.episodes.length;
  $('statUpdates').textContent = trainer.steps.length;

  const ep = currentEpisode();
  $('statEpsilon').textContent = ep ? fmt2(ep.epsilon) : '—';
  $('statOutcome').textContent = ep ? outcomeText(ep) : '—';
  $('statRandom').textContent = ep
    ? `${episodeSteps(ep).filter((st) => !st.greedy).length} of ${ep.count}`
    : '—';

  const recent = trainer.episodes.slice(-20);
  $('statSuccess').textContent = recent.length
    ? `${Math.round((recent.filter((e) => e.outcome === 'goal').length / recent.length) * 100)}%`
    : '—';

  // A trap reached on an untied greedy pick is the only case where the table
  // actually rated the trap best. Exploration and tie-breaks do not count.
  let traps = 0;
  let preferred = 0;
  for (const st of trainer.steps) {
    if (st.reward === -1) { traps++; if (st.greedy && st.ties === 1) preferred++; }
  }
  $('statTraps').textContent = trainer.steps.length
    ? (traps === 0 ? '0' : `${traps} · ${preferred === 0 ? 'none preferred' : `${preferred} preferred`}`)
    : '—';
}

function renderPlayback() {
  const training = mode === 'training';
  $('trainingControls').hidden = !training;
  $('executionControls').hidden = training;
  $('trainBar').hidden = !training;   // execution mode does no learning

  if (training) {
    const n = trainer.episodes.length;
    const epRange = $('episodeRange');
    epRange.max = Math.max(0, n - 1);
    epRange.value = pos.ep;
    epRange.disabled = n === 0;

    const ep = currentEpisode();
    const count = ep ? ep.count : 0;
    const stepRange = $('stepRange');
    stepRange.max = count - 1;
    stepRange.value = pos.step;
    stepRange.disabled = !ep;

    $('episodeLabel').textContent = n ? `${pos.ep + 1} of ${n}` : '—';
    $('stepLabel').textContent = ep
      ? (pos.step < 0 ? `start of ${count}` : `${pos.step + 1} of ${count}`)
      : '—';

    $('epPrev').disabled = !n || pos.ep === 0;
    $('epNext').disabled = !n || pos.ep >= n - 1;
    $('stepPrev').disabled = !ep || pos.step < 0;
    $('stepNext').disabled = !ep || pos.step >= count - 1;
  } else {
    const n = exec.path.length;
    const range = $('execRange');
    range.max = n - 1;
    range.value = exec.idx;
    range.disabled = n === 0;

    $('execStepLabel').textContent = n
      ? (exec.idx < 0 ? `start of ${n}` : `${exec.idx + 1} of ${n}`)
      : '—';
    $('execPrev').disabled = !n || exec.idx < 0;
    $('execNext').disabled = !n || exec.idx >= n - 1;

    const outcomes = {
      goal: 'reaches the <b>goal</b>',
      trap: 'walks into a <b>trap</b>',
      loop: 'gets stuck in a <b>loop</b>',
      timeout: '<b>runs out of steps</b>',
      stuck: 'reaches a cell with <b>no legal moves</b>',
    };
    $('execNote').innerHTML = hasHistory()
      ? `Greedy policy using the Q-table as of ${pos.step < 0
          ? `the start of <b>episode ${pos.ep + 1}</b>`
          : `<b>episode ${pos.ep + 1}, step ${pos.step + 1}</b>`}.
         From the start it ${outcomes[exec.outcome] || '—'} in ${n} step${n === 1 ? '' : 's'}.
         Scrub the training sliders, then return here to compare.`
      : `Nothing has been learned yet, so every Q-value is 0 and the greedy policy is arbitrary.
         Train some episodes first.`;
  }
}

function buildLayoutSelect() {
  const select = $('layoutSelect');
  select.replaceChildren();
  for (const opt of LAYOUTS) {
    const option = document.createElement('option');
    option.value = opt.id;
    option.textContent = opt.name;
    select.appendChild(option);
  }
  select.addEventListener('change', () => {
    switchLayout(LAYOUTS.find((o) => o.id === select.value));
  });
}

function renderLayoutChoice() {
  $('layoutSelect').value = layout.id;
  $('layoutBlurb').textContent = layout.blurb;
  $('worldTitle').textContent = layout.name;
}

/* ------------------------------------------------------------- actions */

function recomputeExecution() {
  const { q } = snapshot();
  const result = trainer.greedyRollout(q, 60);
  exec = { path: result.path, outcome: result.outcome, idx: -1 };
}

function render() {
  renderWorld();
  renderChoicePanel();
  renderQTable();
  renderFormula();
  renderStats();
  renderPlayback();
}

function clampPos() {
  const n = trainer.episodes.length;
  if (!n) { pos = { ep: 0, step: -1 }; return; }
  pos.ep = Math.max(0, Math.min(pos.ep, n - 1));
  pos.step = Math.max(-1, Math.min(pos.step, trainer.episodes[pos.ep].count - 1));
}

function train(n) {
  trainer.runEpisodes(n);
  const last = trainer.episodes.length - 1;
  // Land before the first update so the episode can be stepped through from the top.
  pos = { ep: last, step: -1 };
  if (mode === 'execution') recomputeExecution();
  render();
}

function switchLayout(opt) {
  layout = opt;
  world = new GridWorld(layout, { stepPenalty: effectiveParams().stepPenalty });
  trainer.setWorld(world);          // keep the Q-table on purpose
  if (mode === 'execution') recomputeExecution();
  renderLayoutChoice();
  render();
}

function resetQ() {
  trainer.params = effectiveParams();
  trainer.resetQ();
  pos = { ep: 0, step: -1 };
  exec = { path: [], outcome: null, idx: -1 };
  render();
}

function setMode(next) {
  if (mode === next) return;
  mode = next;
  for (const btn of document.querySelectorAll('.mode')) {
    btn.classList.toggle('is-active', btn.dataset.mode === next);
  }
  if (mode === 'execution') recomputeExecution();
  render();
}

/* step navigation ------------------------------------------------------ */

/* Stepping stays inside the current episode -- use the episode controls to
   move between episodes. */

function stepForward() {
  if (mode === 'execution') {
    if (exec.idx >= exec.path.length - 1) return false;
    exec.idx += 1;
    return true;
  }
  const ep = currentEpisode();
  if (!ep || pos.step >= ep.count - 1) return false;
  pos.step += 1;
  return true;
}

function stepBack() {
  if (mode === 'execution') {
    if (exec.idx < 0) return false;
    exec.idx -= 1;
    return true;
  }
  if (!currentEpisode() || pos.step < 0) return false;
  pos.step -= 1;
  return true;
}

function goToEpisode(i) {
  pos.ep = Math.max(0, Math.min(i, trainer.episodes.length - 1));
  pos.step = -1;
  clampPos();
}

/* ---------------------------------------------------------------- wire */

function afterScrub() {
  if (mode === 'execution') recomputeExecution();
  render();
}

function bindSlider(id, onInput) {
  $(id).addEventListener('input', (e) => onInput(Number(e.target.value)));
}

/** Outputs always show the value the algorithm uses, not the parked slider value. */
function renderParams() {
  const eff = effectiveParams();
  const rows = [
    ['gamma', 'rowGamma', ['gamma']],
    ['epsilon', 'rowEpsilon', ['epsStart', 'epsDecay']],
    ['stepPenalty', 'rowPenalty', ['stepPenalty']],
  ];
  for (const [key, rowId, sliders] of rows) {
    $(rowId).classList.toggle('is-off', !enabled[key]);
    for (const id of sliders) $(id).disabled = !enabled[key];
  }
  $('alphaOut').textContent = fmt2(eff.alpha);
  $('gammaOut').textContent = fmt2(eff.gamma);
  $('epsStartOut').textContent = fmt2(eff.epsilonStart);
  $('epsDecayOut').textContent = fmt2(eff.epsilonDecay);
  $('stepPenaltyOut').textContent = fmt2(eff.stepPenalty);
}

/** Params are baked into the recorded history, so any change starts over. */
function applyParamChange() {
  world = new GridWorld(layout, { stepPenalty: effectiveParams().stepPenalty });
  trainer.setWorld(world);
  renderParams();
  resetQ();
}

function bindParam(id, key) {
  const input = $(id);
  input.value = params[key];
  input.addEventListener('input', () => {
    params[key] = Number(input.value);
    renderParams();
  });
  input.addEventListener('change', applyParamChange);
}

function bindToggle(id, key) {
  const box = $(id);
  box.checked = enabled[key];
  box.addEventListener('change', () => {
    enabled[key] = box.checked;
    applyParamChange();
  });
}

function init() {
  buildLayoutSelect();
  renderLayoutChoice();

  for (const btn of document.querySelectorAll('.mode')) {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  }
  for (const btn of document.querySelectorAll('[data-train]')) {
    btn.addEventListener('click', () => train(Number(btn.dataset.train)));
  }
  $('resetQ').addEventListener('click', resetQ);

  bindParam('alpha', 'alpha');
  bindParam('gamma', 'gamma');
  bindParam('epsStart', 'epsilonStart');
  bindParam('epsDecay', 'epsilonDecay');
  bindParam('stepPenalty', 'stepPenalty');

  bindToggle('useGamma', 'gamma');
  bindToggle('useEpsilon', 'epsilon');
  bindToggle('usePenalty', 'stepPenalty');
  renderParams();

  $('showBars').addEventListener('change', renderWorld);
  $('showPolicy').addEventListener('change', renderWorld);

  $('epPrev').addEventListener('click', () => { goToEpisode(pos.ep - 1); afterScrub(); });
  $('epNext').addEventListener('click', () => { goToEpisode(pos.ep + 1); afterScrub(); });
  $('stepPrev').addEventListener('click', () => { stepBack(); afterScrub(); });
  $('stepNext').addEventListener('click', () => { stepForward(); afterScrub(); });

  bindSlider('episodeRange', (v) => { goToEpisode(v); afterScrub(); });
  bindSlider('stepRange', (v) => { pos.step = v; clampPos(); afterScrub(); });

  $('execPrev').addEventListener('click', () => { stepBack(); render(); });
  $('execNext').addEventListener('click', () => { stepForward(); render(); });
  bindSlider('execRange', (v) => { exec.idx = v; render(); });

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight') { stepForward(); afterScrub(); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { stepBack(); afterScrub(); e.preventDefault(); }
  });

  render();
}

init();
