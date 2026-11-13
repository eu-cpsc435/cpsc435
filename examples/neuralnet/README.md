# Neural Network Playground

A small fully-connected network you can build, step through and train in the browser.
Open `index.html` — no build step, no server, no dependencies.

## What is on screen

- **Forward pass** — the current data point enters on the left, each neuron shows its
  activation, and the prediction sits next to the dashed target. Use the `−` / `+` under a
  column to add or remove neurons, and *Hidden layers* to add or remove a layer. Click a
  neuron to see its linear combination written out with the real numbers.
- **Backward pass** — the same network, showing the gradient instead. Edge thickness is the
  size of ∂L/∂w, colour is the direction the weight is about to move. Neurons show ∂L/∂z.
- **Training data** — every row with its target (dashed ring) and the current prediction
  (dot). Click a row to make it the current point; click a target to flip it.
- **Loss surface** — loss as a function of two weights, **A** and **B**, with every other
  weight held still. Click any edge (or bias stub) to put it on an axis. The dot is where the
  network is, the arrow is downhill, the line is the path taken. Drag to rotate; drag the
  sliders to move a weight by hand.
- **Loss over time** — mean loss over all the data after each step.

## Things to try

1. **Step** a few times on the default network and watch one weight change at a time.
2. Switch *Loss on* between **This point** (stochastic gradient descent — the surface changes
   with every point) and **All data** (batch gradient descent — one stable surface).
3. Pick an input that is 0: its weight's gradient is exactly 0, and the surface is flat along it.
4. Set the task to **XOR** and run. With no hidden layer the loss stalls; add one and it solves it.
5. Push the learning rate up until the dot overshoots the valley.

## Files

| File | Contents |
| --- | --- |
| `js/nn.js` | The network: forward pass, loss, backpropagation, the update rule. No DOM. |
| `js/app.js` | Drawing and controls. |
