const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appSource = fs.readFileSync(path.join(__dirname, '..', 'app.js'), 'utf8');

function createAppContext(seed = 12345) {
  let randomState = seed >>> 0;
  const seededMath = Object.create(Math);
  seededMath.random = () => {
    randomState = (1664525 * randomState + 1013904223) >>> 0;
    return randomState / 4294967296;
  };

  const elements = new Map();
  const context2d = {
    clearRect() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    fillText() {},
    fillRect() {}
  };
  const canvases = new Map();

  const sandbox = {
    Math: seededMath,
    document: {
      getElementById(id) {
        if (id === 'curveCanvas' || id === 'actCanvas') {
          if (!canvases.has(id)) {
            canvases.set(id, {
              width: id === 'curveCanvas' ? 560 : 300,
              height: 380,
              getContext: () => context2d
            });
          }
          return canvases.get(id);
        }

        if (!elements.has(id)) {
          elements.set(id, {
            textContent: '',
            disabled: false,
            addEventListener() {}
          });
        }
        return elements.get(id);
      }
    },
    performance: { now: () => 0 },
    requestAnimationFrame() {},
    console
  };

  vm.createContext(sandbox);
  vm.runInContext(appSource, sandbox);

  return {
    evaluate(expression) {
      return vm.runInContext(expression, sandbox);
    }
  };
}

function snapshot(context, expression) {
  return JSON.parse(context.evaluate(`JSON.stringify(${expression})`));
}

test('activation-aware optimizer training', async t => {
  await t.test('Adam applies bias-corrected updates', () => {
    const context = createAppContext();
    const updated = context.evaluate(`(() => {
      optimizerState = createOptimizerState();
      optimizerState.step = 1;
      return updateParameter(1, 2, 'b3', [], 0.1, 'adam', 0);
    })()`);

    assert.ok(Math.abs(updated - 0.9) < 1e-8);
  });

  await t.test('Adam updates both weights and adaptive activation choices', () => {
    const context = createAppContext();
    const before = snapshot(context, '({ w1, actLogits1, w1Moments: optimizerState.firstMoment.w1, actMoments: optimizerState.firstMoment.actLogits1 })');

    context.evaluate("trainStep(0.73, 0.003, 0.02, null, 'adam')");

    const after = snapshot(context, '({ w1, actLogits1, w1Moments: optimizerState.firstMoment.w1, actMoments: optimizerState.firstMoment.actLogits1 })');
    assert.notDeepEqual(after.w1, before.w1, 'network weights should update');
    assert.notDeepEqual(after.actLogits1, before.actLogits1, 'activation preferences should update');
    assert.ok(after.w1Moments.some(value => value !== 0), 'Adam should track weight gradients');
    assert.ok(after.actMoments.flat().some(value => value !== 0), 'Adam should track activation gradients');
    assert.ok(after.w1.every(Number.isFinite));
    assert.ok(after.actLogits1.flat().every(Number.isFinite));
  });

  await t.test('fixed activation variants update weights without changing activation logits', async t => {
    for (let activationIndex = 0; activationIndex < 4; activationIndex++) {
      await t.test(`fixed ${['Tanh', 'ReLU', 'Sigmoid', 'Sin'][activationIndex]}`, () => {
        const context = createAppContext();
        const before = snapshot(context, '({ w1, actLogits1, actLogits2 })');

        context.evaluate(`trainStep(0.73, 0.003, 0, ${activationIndex}, 'adam')`);

        const after = snapshot(context, '({ w1, actLogits1, actLogits2 })');
        assert.notDeepEqual(after.w1, before.w1, 'network weights should update');
        assert.deepEqual(after.actLogits1, before.actLogits1);
        assert.deepEqual(after.actLogits2, before.actLogits2);
        assert.ok(Number.isFinite(context.evaluate(`forward(0.73, ${activationIndex}).out`)));
      });
    }
  });

  await t.test('SGD, momentum, and Adam stay finite while training', async t => {
    for (const optimizerType of ['sgd', 'momentum', 'adam']) {
      await t.test(optimizerType, () => {
        const context = createAppContext();
        context.evaluate(`for (let i = 0; i < 250; i++) {
          trainStep(0.73, 0.003, 0.02, null, '${optimizerType}');
        }`);

        const result = snapshot(context, '({ w1, w2, w3, actLogits1, actLogits2, loss: calculateEvaluationMSE(), optimizerState })');
        assert.ok(Number.isFinite(result.loss));
        assert.ok(result.w1.every(Number.isFinite));
        assert.ok(result.w2.flat().every(Number.isFinite));
        assert.ok(result.w3.every(Number.isFinite));
        assert.ok(result.actLogits1.flat().every(Number.isFinite));
        assert.ok(result.actLogits2.flat().every(Number.isFinite));
      });
    }
  });
});
