/**
 * gridworld.js -- the environment.
 *
 * A grid of cells. The agent occupies one cell (the "state") and picks one of
 * the actions available there -- moves that would leave the grid or enter a wall
 * are not actions at all. Reward is collected when *entering* a cell:
 *
 *    goal cell (green)      -> +1, episode ends
 *    trap cell (red)        -> -1, episode ends
 *    ordinary cell (gray)   ->  0 (plus an optional step penalty)
 */

const ACTIONS = [
  { name: 'Up',    symbol: '\u2191', dc:  0, dr: -1 },
  { name: 'Right', symbol: '\u2192', dc:  1, dr:  0 },
  { name: 'Down',  symbol: '\u2193', dc:  0, dr:  1 },
  { name: 'Left',  symbol: '\u2190', dc: -1, dr:  0 },
];

const NUM_ACTIONS = ACTIONS.length;

// Cell codes: '.' empty  'G' goal (+1)  'P' punishment (-1)  '#' wall  'S' start
const LAYOUTS = [
  {
    id: 'classic',
    name: 'Classic',
    blurb: 'A goal in the top row with two traps in the way.',
    rows: [
      '..G.',
      '.P..',
      'S.P.',
    ],
  },
  {
    id: 'shifted',
    name: 'Shifted Goal',
    blurb: 'Same start, goal moved to the opposite corner. A Q-table trained on "Classic" is now confidently wrong.',
    rows: [
      '.P..',
      '..P.',
      'S..G',
    ],
  },
  {
    id: 'cliff',
    name: 'Cliff Walk',
    blurb: 'The goal is three steps away, along a row of traps.',
    rows: [
      '....',
      '....',
      'SPPG',
    ],
  },
  {
    id: 'maze',
    name: 'Maze',
    blurb: 'A wall forces a detour, and a trap sits right beside the goal.',
    rows: [
      'S#.G',
      '.#.P',
      '....',
    ],
  },
];

class GridWorld {
  constructor(layout, { stepPenalty = 0 } = {}) {
    this.layout = layout;
    this.rows = layout.rows.length;
    this.cols = layout.rows[0].length;
    this.stepPenalty = stepPenalty;

    this.cells = [];
    this.start = 0;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const ch = layout.rows[r][c];
        this.cells.push(ch === 'S' ? '.' : ch);
        if (ch === 'S') this.start = this.stateOf(c, r);
      }
    }

    this._valid = [];
    for (let s = 0; s < this.numStates; s++) this._valid.push(this._computeValid(s));
  }

  /**
   * Actions that actually lead somewhere. Walking off the edge or into a wall is
   * not a move the agent can make, so those actions are excluded entirely: they
   * are never selected, never given a Q-value, and never part of max Q(s',.).
   * Terminal cells have no actions at all, since the episode ends on arrival.
   */
  _computeValid(s) {
    if (this.isWall(s) || this.isTerminal(s)) return [];
    const out = [];
    for (let a = 0; a < NUM_ACTIONS; a++) {
      const c = this.colOf(s) + ACTIONS[a].dc;
      const r = this.rowOf(s) + ACTIONS[a].dr;
      if (c < 0 || c >= this.cols || r < 0 || r >= this.rows) continue;
      if (this.isWall(this.stateOf(c, r))) continue;
      out.push(a);
    }
    return out;
  }

  validActions(s) { return this._valid[s]; }
  isValidAction(s, a) { return this._valid[s].includes(a); }

  get numStates() { return this.rows * this.cols; }

  stateOf(c, r) { return r * this.cols + c; }
  colOf(s) { return s % this.cols; }
  rowOf(s) { return Math.floor(s / this.cols); }

  cellAt(s) { return this.cells[s]; }
  isWall(s) { return this.cells[s] === '#'; }
  isGoal(s) { return this.cells[s] === 'G'; }
  isTrap(s) { return this.cells[s] === 'P'; }
  isTerminal(s) { return this.isGoal(s) || this.isTrap(s); }

  /** Cells the agent can actually occupy and learn about. */
  activeStates() {
    const out = [];
    for (let s = 0; s < this.numStates; s++) if (!this.isWall(s)) out.push(s);
    return out;
  }

  /**
   * Apply an action. Deterministic: the agent always moves where it intends.
   * Only valid actions are ever selected, so the blocked branch below is a
   * safety net rather than part of the model.
   */
  step(s, a) {
    const { dc, dr } = ACTIONS[a];
    const c = this.colOf(s) + dc;
    const r = this.rowOf(s) + dr;

    let next = s;
    if (c >= 0 && c < this.cols && r >= 0 && r < this.rows) {
      const candidate = this.stateOf(c, r);
      if (!this.isWall(candidate)) next = candidate;
    }

    if (this.isGoal(next))  return { next, reward:  1, done: true };
    if (this.isTrap(next))  return { next, reward: -1, done: true };
    return { next, reward: this.stepPenalty, done: false };
  }
}
