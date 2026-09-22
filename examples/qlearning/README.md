# Q-Learning Grid World

An interactive demo of tabular Q-learning. The agent (yellow circle) moves around a
4x3 grid looking for the green goal (+1) while avoiding red traps (-1). Gray cells
give 0.

Everything the agent knows lives in one table of numbers, `Q[state][action]`, and that
table is built by applying a single rule once per step:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

## Running it

Open `index.html` in a browser. That is all -- no build step and no server.

The scripts are plain `<script>` tags rather than ES modules for exactly this reason:
browsers block module loading over `file://`, which would leave the page rendered but
dead (empty Q-table, buttons that do nothing). If you would rather serve it, any static
server works:

```bash
py -m http.server 8321
```

## What to look at

**The bars on each cell.** Every cell has up to four bars, one per edge, showing the
Q-value for moving in that direction. Green means the value is zero or positive, red
means negative, and thickness scales with magnitude. An edge with no bar is an action
the agent has never tried — that is the difference between "known to be bad" and
"unknown", which the numbers alone make easy to miss.

**The choice compass.** Beside the grid, the four directions are drawn where they point,
each showing the value the agent is reading right now. It is always about the cell the
agent currently occupies and the move it is **about to make** -- at the start of an
episode it already shows the first decision, before anything has happened:

- a **dashed arm with a dash** is not a legal move from that cell at all;
- an **outlined arm** holds the best value -- what greedy would pick. Several can be
  outlined at once, which is a tie;
- the **filled arm** is what the agent actually took.

When the filled arm is also outlined, the agent followed the table. When a *different*
arm is outlined, it explored: it could see the better value and ignored it. That single
visual is the whole of epsilon-greedy, and it is worth pausing on -- in execution mode
the filled arm is always the outlined one, because nothing is left to chance.

**The Result panel under the timelines.** The counterpart to the compass: the compass
shows the move about to be made, Result reports the move just made and the update it
caused. Step through a training run and it substitutes real numbers into the formula. Each symbol sits directly above
its own value in a shared column, so there is no guessing about which number is which
term, and hovering (or tabbing to) any term pops up a description of what it represents.
Below the formula is a compact row of the rest of the step: state, action, next state,
TD target and TD error. The interesting moments are:

- the first step that touches a terminal cell, where `max Q(s',.) = 0` and the target
  collapses to the raw reward — this is where value first enters the table;
- any later step on a gray cell, where `r = 0` and the entire target comes from
  `gamma * max Q(s',.)` — this is value flowing one cell backwards per visit;
- a step where the TD error is 0, which is what convergence looks like.

**Switching terms on one at a time.** With everything off the rule reduces to
`Q <- Q + alpha [ r + max Q(s',.) - Q ]`, which still solves every layout here. Turn the
terms on to show what each one buys:

- **gamma.** Off means gamma = 1, and every action on a path to the goal converges to the
  same 1.000 -- the bars become a flat wall of green and the numbers say nothing about
  distance. Switch it on and value decays by a factor of gamma per step away from the
  goal. On "Classic" the optimal path then reads 0.729, 0.810, 0.900, 1.000, which is
  exactly 0.9^3 down to 0.9^0. That gradient is what the bar thicknesses are showing.
- **epsilon.** Worth being honest about: on grids this small, exploration is *not* what
  finds the optimal path. Every action starts tied at 0.000, so the random tie-break
  already walks the agent over the whole grid. Measured across 40 seeds x 200 episodes,
  the greedy path comes out optimal on all four layouts with epsilon off. What switching
  it on visibly does here is add trap entries and cost a little goal rate, since ~5% of
  moves stay random forever. Exploration earns its keep in larger or stochastic problems,
  where the opening tie-break is not enough coverage -- which is itself a useful thing to
  be able to show rather than assert.
- **step penalty.** Makes wandering cost something directly, rather than only costing
  discounted reward.

**No bar means the action was never tried; no bar *and* a dash in the table means the
action does not exist there at all.** Worth separating those two in discussion: one is
missing knowledge, the other is a property of the world.

**Why the agent keeps hitting traps.** It looks like the agent is not learning, but it
is: it never *prefers* a trap. One visit settles the question -- the action drops from
0.000 to -0.200, already worse than every alternative, and later visits only refine the
magnitude toward -1. Trap entries after that come from exploration, or from a tie-break
back when the whole row was still 0.000. With epsilon on, early training looks random
because it *is* random: epsilon starts at 1.0 and decays 0.95 per episode, so the first
~20 episodes barely consult the table, and the 0.05 floor keeps ~5% of moves random
forever. The Progress panel spells this out with "Random moves this episode", an outcome
of "trap, exploring" vs "trap, chose it", and a running "Traps entered / none preferred".
The choice compass names the cause of each individual step, including how many actions
were tied when the pick was greedy.

**Training vs. Execution.** Training mode explores (epsilon-greedy) and updates the
table. Execution mode turns both off and just follows `argmax_a Q(s,a)` from the start.
The execution rollout uses the Q-table *as of the training step you have scrubbed to*,
so you can move the episode slider back and watch the policy get worse.

**Switching worlds.** The four layouts in the header dropdown all use the same grid
size, so the Q-table stays meaningful across them, and switching keeps
the learned table on purpose. Train to convergence on "Classic", then switch to
"Shifted Goal" and look at Execution mode: the agent confidently walks into a trap. The
Q-table is indexed by cell number and nothing else, so it has no notion that the world
changed, and no way to transfer what it learned about "move away from red" to a cell it
has not personally visited. That limitation is the motivation for function approximation
and deep Q-learning.

## Controls

- **+1 / +10 / +100 episodes** — run training and append to the recorded history.
  Training mode only; execution mode does no learning, so the buttons are hidden there.
- **Episode / Step sliders** — scrub anywhere in that history. Nothing is re-simulated;
  the recorded updates are replayed, so scrubbing backwards is exact. Stepping stays
  inside the current episode; use the episode control to move between episodes.
- **Arrow keys** step back and forward.
- **Hyperparameters** — gamma, epsilon and the step penalty each have a checkbox and
  start **off**, so the demo opens on the plainest form of the rule. Unchecked means the
  term has no effect (gamma = 1, epsilon = 0, penalty = 0) and the readout shows that
  neutral value, not the parked slider value. Checking one activates its slider. Changing
  anything clears training, since the recorded history was produced under the old values.
- **Reset Q-table** — zero everything, keep the current world.

## Files

| File | Contents |
| --- | --- |
| `js/gridworld.js` | The environment: cells, layouts, rewards, transitions. |
| `js/qlearning.js` | The algorithm: epsilon-greedy action selection, the update rule, episode recording and replay. |
| `js/app.js` | Rendering and controls. |
| `styles.css` | Styling. |

`gridworld.js` and `qlearning.js` have no DOM dependencies and are the two files worth
reading to understand the method.

## Notes on the implementation

- Movement is deterministic, and only *valid* actions exist. An action that would leave
  the grid or walk into a wall is excluded entirely from that cell: it is never selected,
  never given a Q-value, and never part of `max Q(s',.)`. So the bottom-left cell has two
  actions, not four, and the Q-table shows a dash for the rest. Terminal cells have no
  actions at all.
- Rewards are collected on *entering* a cell. Goal and trap cells are terminal.
- Exploration decays per episode: `epsilon = max(epsilon_min, epsilon_0 * decay^episode)`.
- The RNG is seeded, so a reset followed by the same number of episodes reproduces the
  same run — useful when demonstrating live.
- Training history is stored as a flat list of updates. Any point in that history is
  reconstructed by replaying the updates up to it, which keeps memory small and makes
  the episode/step sliders exact rather than approximate.
