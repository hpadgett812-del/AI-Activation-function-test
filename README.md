# AI-Activation-function-test

The demo is separated into `index.html` for the page, `styles.css` for presentation, and `app.js` for training and visualization. Open `index.html` directly in a modern browser; no build step or external dependencies are required.

Run the optimizer subtests with `node --test tests/optimizer.test.js`.

The network learns activation preferences while fitting a Gaussian bell curve. Backpropagation updates the network weights and activation-choice logits together on every training step.

Start the AI population to train eight models side by side on the same samples. They vary activation choices, learning rates, and SGD, momentum, or Adam. Adam uses bias-corrected first and second gradient moments; it often converges quickly, but is not guaranteed to win this task. Models that reach the evaluation-MSE target are ranked by elapsed time to reach it, with training steps shown; models that have not reached it are ordered by their best evaluation MSE.

At the end of each generation, the top models are copied and mutated to create the next generation. This is a small evolutionary experiment, not a claim that one activation or optimizer is universally best.
