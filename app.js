    const curveCanvas = document.getElementById('curveCanvas');
    const cCtx = curveCanvas.getContext('2d');
    const actCanvas = document.getElementById('actCanvas');
    const aCtx = actCanvas.getContext('2d');

    const toggleBtn = document.getElementById('toggleBtn');
    const resetBtn = document.getElementById('resetBtn');
    const compareBtn = document.getElementById('compareBtn');
    const epochDisplay = document.getElementById('epochDisplay');
    const trainingLossDisplay = document.getElementById('trainingLossDisplay');
    const evaluationLossDisplay = document.getElementById('evaluationLossDisplay');
    const comparisonStatus = document.getElementById('comparisonStatus');
    const comparisonResults = document.getElementById('comparisonResults');

    const LEARNING_RATE = 0.01;
    const ACT_LEARNING_RATE = 0.08;
    const SOFTMAX_TEMP = 0.38;
    const ENTROPY_PENALTY = 0.002;
    const STEPS_PER_FRAME = 25;
    const HIDDEN_SIZE = 16;
    const TRAINING_SAMPLE_COUNT = 161;
    const EVALUATION_SAMPLE_COUNT = 160;
    const POPULATION_SIZE = 8;
    const POPULATION_STEPS_PER_FRAME = 2;
    const GENERATION_STEPS = 800;
    const EVALUATION_INTERVAL = 25;
    const TARGET_EVALUATION_MSE = 0.02;
    const ADAM_BETA1 = 0.9;
    const ADAM_BETA2 = 0.999;
    const ADAM_EPSILON = 1e-8;

    const CANDIDATES = [
      { name: 'Tanh', color: '#ab47bc', fn: z => Math.tanh(z), dfn: z => 1 - Math.tanh(z) * Math.tanh(z) },
      { name: 'ReLU', color: '#ffa726', fn: z => Math.max(0, z), dfn: z => z > 0 ? 1 : 0 },
      { name: 'Sigmoid', color: '#29b6f6', fn: z => 1 / (1 + Math.exp(-z)), dfn: z => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } },
      { name: 'Sin', color: '#26a69a', fn: z => Math.sin(z), dfn: z => Math.cos(z) }
    ];
    const NUM_ACTS = CANDIDATES.length;

    let w1, b1, actLogits1;
    let w2, b2, actLogits2;
    let w3, b3;
    let epoch = 0;
    let isRunning = true;
    let trainingLoss = null;
    let optimizerState;
    let population = null;
    let populationRunning = false;
    let populationStep = 0;
    let generation = 0;
    let populationRandom = Math.random;

    function bellCurve(x) {
      return Math.exp(-0.5 * x * x);
    }

    function softmax(logits, temp = SOFTMAX_TEMP) {
      const scaled = logits.map(v => v / temp);
      const max = Math.max(...scaled);
      const exps = scaled.map(v => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(v => v / sum);
    }

    function initNetwork() {
      epoch = 0;
      trainingLoss = null;

      w1 = Array.from({ length: HIDDEN_SIZE }, () => (Math.random() - 0.5) * 2);
      b1 = Array.from({ length: HIDDEN_SIZE }, () => (Math.random() - 0.5) * 2);
      actLogits1 = Array.from({ length: HIDDEN_SIZE }, () =>
        Array.from({ length: NUM_ACTS }, () => (Math.random() - 0.5) * 0.1)
      );

      w2 = Array.from({ length: HIDDEN_SIZE }, () =>
        Array.from({ length: HIDDEN_SIZE }, () => (Math.random() - 0.5) * 1)
      );
      b2 = Array.from({ length: HIDDEN_SIZE }, () => 0);
      actLogits2 = Array.from({ length: HIDDEN_SIZE }, () =>
        Array.from({ length: NUM_ACTS }, () => (Math.random() - 0.5) * 0.1)
      );

      w3 = Array.from({ length: HIDDEN_SIZE }, () => (Math.random() - 0.5) * 1);
      b3 = 0;
      optimizerState = createOptimizerState();
    }

    function createZeroParameters() {
      return {
        w1: Array(HIDDEN_SIZE).fill(0),
        b1: Array(HIDDEN_SIZE).fill(0),
        w2: Array.from({ length: HIDDEN_SIZE }, () => Array(HIDDEN_SIZE).fill(0)),
        b2: Array(HIDDEN_SIZE).fill(0),
        w3: Array(HIDDEN_SIZE).fill(0),
        b3: 0,
        actLogits1: Array.from({ length: HIDDEN_SIZE }, () => Array(NUM_ACTS).fill(0)),
        actLogits2: Array.from({ length: HIDDEN_SIZE }, () => Array(NUM_ACTS).fill(0))
      };
    }

    function createOptimizerState() {
      return {
        velocity: createZeroParameters(),
        firstMoment: createZeroParameters(),
        secondMoment: createZeroParameters(),
        step: 0
      };
    }

    function updateParameter(value, gradient, key, indices, learningRate, optimizerType, momentum) {
      const readSlot = slot => indices.reduce((current, index) => current[index], slot[key]);
      const writeSlot = (slot, nextValue) => {
        let current = slot[key];
        for (let i = 0; i < indices.length - 1; i++) {
          current = current[indices[i]];
        }
        if (indices.length === 0) slot[key] = nextValue;
        else current[indices[indices.length - 1]] = nextValue;
      };

      if (optimizerType === 'adam') {
        const firstMoment = ADAM_BETA1 * readSlot(optimizerState.firstMoment)
          + (1 - ADAM_BETA1) * gradient;
        const secondMoment = ADAM_BETA2 * readSlot(optimizerState.secondMoment)
          + (1 - ADAM_BETA2) * gradient * gradient;
        writeSlot(optimizerState.firstMoment, firstMoment);
        writeSlot(optimizerState.secondMoment, secondMoment);

        const correctedFirstMoment = firstMoment / (1 - ADAM_BETA1 ** optimizerState.step);
        const correctedSecondMoment = secondMoment / (1 - ADAM_BETA2 ** optimizerState.step);
        return value - learningRate * correctedFirstMoment
          / (Math.sqrt(correctedSecondMoment) + ADAM_EPSILON);
      }

      const velocity = momentum * readSlot(optimizerState.velocity) + gradient;
      writeSlot(optimizerState.velocity, velocity);
      return value - learningRate * velocity;
    }

    function forward(x, fixedActivationIndex = null) {
      const z1 = new Array(HIDDEN_SIZE);
      const probs1 = new Array(HIDDEN_SIZE);
      const candOut1 = new Array(HIDDEN_SIZE);
      const h1 = new Array(HIDDEN_SIZE);

      for (let i = 0; i < HIDDEN_SIZE; i++) {
        z1[i] = x * w1[i] + b1[i];
        probs1[i] = fixedActivationIndex === null
          ? softmax(actLogits1[i])
          : CANDIDATES.map((_, index) => Number(index === fixedActivationIndex));
        candOut1[i] = CANDIDATES.map(candidate => candidate.fn(z1[i]));
        let sum = 0;
        for (let k = 0; k < NUM_ACTS; k++) {
          sum += probs1[i][k] * candOut1[i][k];
        }
        h1[i] = sum;
      }

      const z2 = new Array(HIDDEN_SIZE);
      const probs2 = new Array(HIDDEN_SIZE);
      const candOut2 = new Array(HIDDEN_SIZE);
      const h2 = new Array(HIDDEN_SIZE);

      for (let j = 0; j < HIDDEN_SIZE; j++) {
        let sumZ = b2[j];
        for (let i = 0; i < HIDDEN_SIZE; i++) {
          sumZ += h1[i] * w2[j][i];
        }
        z2[j] = sumZ;
        probs2[j] = fixedActivationIndex === null
          ? softmax(actLogits2[j])
          : CANDIDATES.map((_, index) => Number(index === fixedActivationIndex));
        candOut2[j] = CANDIDATES.map(candidate => candidate.fn(z2[j]));
        let sumH = 0;
        for (let k = 0; k < NUM_ACTS; k++) {
          sumH += probs2[j][k] * candOut2[j][k];
        }
        h2[j] = sumH;
      }

      let out = b3;
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        out += h2[j] * w3[j];
      }

      return { z1, probs1, candOut1, h1, z2, probs2, candOut2, h2, out };
    }

    function trainStep(
      x = -4 + (8 * Math.floor(Math.random() * TRAINING_SAMPLE_COUNT)) / (TRAINING_SAMPLE_COUNT - 1),
      weightLearningRate = LEARNING_RATE,
      activationLearningRate = ACT_LEARNING_RATE,
      fixedActivationIndex = null,
      optimizerType = 'sgd'
    ) {
      const target = bellCurve(x);
      const { z1, probs1, candOut1, h1, z2, probs2, candOut2, h2, out } = forward(x, fixedActivationIndex);
      const dOut = out - target;
      const sampleLoss = dOut * dOut;
      trainingLoss = trainingLoss === null
        ? sampleLoss
        : trainingLoss * 0.99 + sampleLoss * 0.01;

      const dw3 = new Array(HIDDEN_SIZE);
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        dw3[j] = dOut * h2[j];
      }

      const dz2 = new Array(HIDDEN_SIZE);
      const dLogits2 = Array.from({ length: HIDDEN_SIZE }, () => new Array(NUM_ACTS));

      for (let j = 0; j < HIDDEN_SIZE; j++) {
        const dh2 = dOut * w3[j];
        let dAct_dz = 0;
        for (let k = 0; k < NUM_ACTS; k++) {
          dAct_dz += probs2[j][k] * CANDIDATES[k].dfn(z2[j]);
        }
        dz2[j] = dh2 * dAct_dz;

        const entropy = -probs2[j].reduce((sum, probability) =>
          sum + probability * Math.log(probability + 1e-8), 0);
        for (let k = 0; k < NUM_ACTS; k++) {
          const taskGrad = dh2 * (probs2[j][k] / SOFTMAX_TEMP) * (candOut2[j][k] - h2[j]);
          const entropyGrad = -ENTROPY_PENALTY * probs2[j][k]
            * (Math.log(probs2[j][k] + 1e-8) + entropy) / SOFTMAX_TEMP;
          dLogits2[j][k] = taskGrad + entropyGrad;
        }
      }

      const dz1 = new Array(HIDDEN_SIZE);
      const dLogits1 = Array.from({ length: HIDDEN_SIZE }, () => new Array(NUM_ACTS));

      for (let i = 0; i < HIDDEN_SIZE; i++) {
        let dh1 = 0;
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          dh1 += dz2[j] * w2[j][i];
        }

        let dAct_dz = 0;
        for (let k = 0; k < NUM_ACTS; k++) {
          dAct_dz += probs1[i][k] * CANDIDATES[k].dfn(z1[i]);
        }
        dz1[i] = dh1 * dAct_dz;

        const entropy = -probs1[i].reduce((sum, probability) =>
          sum + probability * Math.log(probability + 1e-8), 0);
        for (let k = 0; k < NUM_ACTS; k++) {
          const taskGrad = dh1 * (probs1[i][k] / SOFTMAX_TEMP) * (candOut1[i][k] - h1[i]);
          const entropyGrad = -ENTROPY_PENALTY * probs1[i][k]
            * (Math.log(probs1[i][k] + 1e-8) + entropy) / SOFTMAX_TEMP;
          dLogits1[i][k] = taskGrad + entropyGrad;
        }
      }

      optimizerState.step += 1;
      b3 = updateParameter(b3, dOut, 'b3', [], weightLearningRate, optimizerType,
        optimizerType === 'momentum' ? 0.8 : 0);

      for (let j = 0; j < HIDDEN_SIZE; j++) {
        w3[j] = updateParameter(w3[j], dw3[j], 'w3', [j], weightLearningRate, optimizerType,
          optimizerType === 'momentum' ? 0.8 : 0);
        b2[j] = updateParameter(b2[j], dz2[j], 'b2', [j], weightLearningRate, optimizerType,
          optimizerType === 'momentum' ? 0.8 : 0);
        for (let i = 0; i < HIDDEN_SIZE; i++) {
          const gradient = dz2[j] * h1[i];
          w2[j][i] = updateParameter(w2[j][i], gradient, 'w2', [j, i],
            weightLearningRate, optimizerType, optimizerType === 'momentum' ? 0.8 : 0);
        }
        if (fixedActivationIndex === null) {
          for (let k = 0; k < NUM_ACTS; k++) {
            actLogits2[j][k] = updateParameter(actLogits2[j][k], dLogits2[j][k],
              'actLogits2', [j, k], activationLearningRate, optimizerType,
              optimizerType === 'momentum' ? 0.8 : 0);
          }
        }
      }

      for (let i = 0; i < HIDDEN_SIZE; i++) {
        const weightGradient = dz1[i] * x;
        w1[i] = updateParameter(w1[i], weightGradient, 'w1', [i], weightLearningRate,
          optimizerType, optimizerType === 'momentum' ? 0.8 : 0);
        b1[i] = updateParameter(b1[i], dz1[i], 'b1', [i], weightLearningRate,
          optimizerType, optimizerType === 'momentum' ? 0.8 : 0);
        if (fixedActivationIndex === null) {
          for (let k = 0; k < NUM_ACTS; k++) {
            actLogits1[i][k] = updateParameter(actLogits1[i][k], dLogits1[i][k],
              'actLogits1', [i, k], activationLearningRate, optimizerType,
              optimizerType === 'momentum' ? 0.8 : 0);
          }
        }
      }

      epoch += 1;
    }

    function copyParameterSet(source) {
      return {
        w1: source.w1.slice(),
        b1: source.b1.slice(),
        w2: source.w2.map(row => row.slice()),
        b2: source.b2.slice(),
        w3: source.w3.slice(),
        b3: source.b3,
        actLogits1: source.actLogits1.map(row => row.slice()),
        actLogits2: source.actLogits2.map(row => row.slice())
      };
    }

    function copyOptimizerState(source) {
      return {
        velocity: copyParameterSet(source.velocity),
        firstMoment: copyParameterSet(source.firstMoment),
        secondMoment: copyParameterSet(source.secondMoment),
        step: source.step
      };
    }

    function snapshotLiveModel() {
      return {
        w1: w1.slice(),
        b1: b1.slice(),
        w2: w2.map(row => row.slice()),
        b2: b2.slice(),
        w3: w3.slice(),
        b3,
        actLogits1: actLogits1.map(row => row.slice()),
        actLogits2: actLogits2.map(row => row.slice()),
        optimizerState: copyOptimizerState(optimizerState),
        epoch: 0,
        trainingLoss: null
      };
    }

    function cloneModelState(source) {
      return {
        w1: source.w1.slice(),
        b1: source.b1.slice(),
        w2: source.w2.map(row => row.slice()),
        b2: source.b2.slice(),
        w3: source.w3.slice(),
        b3: source.b3,
        actLogits1: source.actLogits1.map(row => row.slice()),
        actLogits2: source.actLogits2.map(row => row.slice()),
        optimizerState: createOptimizerState(),
        epoch: 0,
        trainingLoss: null
      };
    }

    function withPopulationModel(individual, operation) {
      const liveState = {
        w1, b1, w2, b2, w3, b3, actLogits1, actLogits2,
        optimizerState, epoch, trainingLoss
      };
      const state = individual.state;

      w1 = state.w1;
      b1 = state.b1;
      w2 = state.w2;
      b2 = state.b2;
      w3 = state.w3;
      b3 = state.b3;
      actLogits1 = state.actLogits1;
      actLogits2 = state.actLogits2;
      optimizerState = state.optimizerState;
      epoch = state.epoch;
      trainingLoss = state.trainingLoss;

      try {
        return operation();
      } finally {
        state.w1 = w1;
        state.b1 = b1;
        state.w2 = w2;
        state.b2 = b2;
        state.w3 = w3;
        state.b3 = b3;
        state.actLogits1 = actLogits1;
        state.actLogits2 = actLogits2;
        state.optimizerState = optimizerState;
        state.epoch = epoch;
        state.trainingLoss = trainingLoss;

        ({ w1, b1, w2, b2, w3, b3, actLogits1, actLogits2,
          optimizerState, epoch, trainingLoss } = liveState);
      }
    }

    function individualDescription(individual) {
      const activation = individual.activationIndex === null
        ? 'adaptive activations'
        : `${CANDIDATES[individual.activationIndex].name} only`;
      const optimizer = individual.optimizerType === 'momentum'
        ? 'Momentum (0.8)'
        : individual.optimizerType.toUpperCase();
      return `${activation}; lr ${individual.learningRate}; ${optimizer}`;
    }

    function createIndividual(id, genome, state) {
      return {
        id,
        parentId: null,
        activationIndex: genome.activationIndex,
        learningRate: genome.learningRate,
        activationLearningRate: genome.activationLearningRate,
        optimizerType: genome.optimizerType,
        state,
        steps: 0,
        stepsToTarget: null,
        timeToTargetMs: null,
        bestMSE: Infinity,
        startedAt: performance.now()
      };
    }

    function startPopulation() {
      const base = snapshotLiveModel();
      const genomes = [
        ...CANDIDATES.map((_, activationIndex) => ({
          activationIndex, learningRate: 0.01, activationLearningRate: 0, optimizerType: 'sgd'
        })),
        { activationIndex: null, learningRate: 0.006, activationLearningRate: 0.04, optimizerType: 'sgd' },
        { activationIndex: null, learningRate: 0.003, activationLearningRate: 0.02, optimizerType: 'adam' },
        { activationIndex: 0, learningRate: 0.004, activationLearningRate: 0, optimizerType: 'momentum' },
        { activationIndex: 3, learningRate: 0.003, activationLearningRate: 0, optimizerType: 'adam' }
      ];

      population = genomes.map((genome, index) =>
        createIndividual(index + 1, genome, cloneModelState(base))
      );
      populationStep = 0;
      generation = 1;
      populationRandom = comparisonRandom(0x71a1b00b);
      for (const individual of population) {
        individual.bestMSE = withPopulationModel(individual, () =>
          calculateEvaluationMSE(x => forward(x, individual.activationIndex))
        );
      }
      populationRunning = true;
      compareBtn.textContent = 'Pause population';
      updatePopulationDisplay();
    }

    function comparisonRandom(seed) {
      let state = seed >>> 0;
      return () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
      };
    }

    function rankPopulation(individuals = population) {
      return [...individuals].sort((a, b) => {
        const aReached = a.stepsToTarget !== null;
        const bReached = b.stepsToTarget !== null;
        if (aReached !== bReached) return aReached ? -1 : 1;
        if (aReached && a.timeToTargetMs !== b.timeToTargetMs) {
          return a.timeToTargetMs - b.timeToTargetMs;
        }
        return a.bestMSE - b.bestMSE || a.stepsToTarget - b.stepsToTarget;
      });
    }

    function mutateGenome(parent) {
      const genome = {
        activationIndex: parent.activationIndex,
        learningRate: parent.learningRate,
        activationLearningRate: parent.activationLearningRate,
        optimizerType: parent.optimizerType
      };
      const mutationCount = genome.activationIndex === null ? 4 : 3;
      const mutation = Math.floor(Math.random() * mutationCount);
      if (mutation === 0) {
        const activationOptions = [null, ...CANDIDATES.map((_, index) => index)]
          .filter(index => index !== genome.activationIndex);
        genome.activationIndex = activationOptions[Math.floor(Math.random() * activationOptions.length)];
      } else if (mutation === 1) {
        const options = [0.003, 0.006, 0.01, 0.018, 0.03].filter(value => value !== genome.learningRate);
        genome.learningRate = options[Math.floor(Math.random() * options.length)];
      } else if (mutation === 2) {
        const options = ['sgd', 'momentum', 'adam'].filter(value => value !== genome.optimizerType);
        genome.optimizerType = options[Math.floor(Math.random() * options.length)];
      } else {
        const options = [0.02, 0.05, 0.1].filter(value => value !== genome.activationLearningRate);
        genome.activationLearningRate = options[Math.floor(Math.random() * options.length)];
      }
      return genome;
    }

    function breedNextGeneration() {
      const ranked = rankPopulation();
      const parents = ranked.slice(0, 4);
      const survivors = ranked.slice(0, 2);
      const next = survivors.slice();

      for (let i = 0; i < POPULATION_SIZE - survivors.length; i++) {
        const parent = parents[Math.floor(Math.random() * parents.length)];
        const child = createIndividual(
          generation * POPULATION_SIZE + next.length + 1,
          mutateGenome(parent),
          cloneModelState(parent.state)
        );
        child.parentId = parent.id;
        next.push(child);
      }

      generation += 1;
      populationStep = 0;
      population = next;
      for (const individual of population) {
        individual.steps = 0;
        individual.stepsToTarget = null;
        individual.timeToTargetMs = null;
        individual.startedAt = performance.now();
        individual.bestMSE = withPopulationModel(individual, () =>
          calculateEvaluationMSE(x => forward(x, individual.activationIndex))
        );
      }
    }

    function updatePopulationDisplay() {
      if (!population) return;
      const ranked = rankPopulation();
      comparisonResults.textContent = ranked.map((individual, index) => {
        const speed = individual.stepsToTarget === null
          ? `not reached (best MSE ${individual.bestMSE.toFixed(5)})`
          : `target in ${(individual.timeToTargetMs / 1000).toFixed(2)}s / ${individual.stepsToTarget} steps`;
        const lineage = individual.parentId === null ? '' : ` ← AI ${individual.parentId}`;
        return `${index + 1}. AI ${individual.id}${lineage} — ${speed}\n   ${individualDescription(individual)}`;
      }).join('\n');
      comparisonStatus.textContent =
        `${populationRunning ? 'Training' : 'Paused'} · generation ${generation} · ` +
        `${populationStep.toLocaleString()} / ${GENERATION_STEPS.toLocaleString()} shared samples per AI · ` +
        `target MSE ≤ ${TARGET_EVALUATION_MSE} (checked every ${EVALUATION_INTERVAL} steps)`;
    }

    function togglePopulation() {
      if (!population) {
        startPopulation();
        return;
      }

      populationRunning = !populationRunning;
      compareBtn.textContent = populationRunning ? 'Pause population' : 'Resume population';
      updatePopulationDisplay();
    }

    function advancePopulation() {
      if (!populationRunning || !population) return;

      for (let step = 0; step < POPULATION_STEPS_PER_FRAME; step++) {
        const sampleIndex = Math.floor(populationRandom() * TRAINING_SAMPLE_COUNT);
        const x = -4 + (8 * sampleIndex) / (TRAINING_SAMPLE_COUNT - 1);
        for (const individual of population) {
          withPopulationModel(individual, () => {
            trainStep(
              x,
              individual.learningRate,
              individual.activationLearningRate,
              individual.activationIndex,
              individual.optimizerType
            );
          });
          individual.steps += 1;
          if (individual.steps % EVALUATION_INTERVAL === 0) {
            const mse = withPopulationModel(individual, () =>
              calculateEvaluationMSE(input => forward(input, individual.activationIndex))
            );
            individual.bestMSE = Math.min(individual.bestMSE, mse);
            if (individual.stepsToTarget === null && mse <= TARGET_EVALUATION_MSE) {
              individual.stepsToTarget = individual.steps;
              individual.timeToTargetMs = performance.now() - individual.startedAt;
            }
          }
        }

        populationStep += 1;
        if (populationStep >= GENERATION_STEPS) {
          breedNextGeneration();
          break;
        }
      }

      updatePopulationDisplay();
    }

    function toCurveX(x) {
      return ((x + 4) / 8) * curveCanvas.width;
    }

    function toCurveY(y) {
      return curveCanvas.height - (y * (curveCanvas.height * 0.75) + 50);
    }

    function calculateEvaluationMSE(forwardPass = forward) {
      let totalLoss = 0;
      for (let i = 0; i < EVALUATION_SAMPLE_COUNT; i++) {
        const x = -4 + (8 * (i + 0.5)) / EVALUATION_SAMPLE_COUNT;
        const error = forwardPass(x).out - bellCurve(x);
        totalLoss += error * error;
      }
      return totalLoss / EVALUATION_SAMPLE_COUNT;
    }

    function drawCurves() {
      cCtx.clearRect(0, 0, curveCanvas.width, curveCanvas.height);

      const curvePoints = [];
      for (let px = 0; px <= curveCanvas.width; px += 2) {
        const x = (px / curveCanvas.width) * 8 - 4;
        curvePoints.push({ px, x, ...forward(x) });
      }

      cCtx.strokeStyle = '#333';
      cCtx.lineWidth = 1;
      cCtx.beginPath();
      cCtx.moveTo(0, toCurveY(0));
      cCtx.lineTo(curveCanvas.width, toCurveY(0));
      cCtx.moveTo(toCurveX(0), 0);
      cCtx.lineTo(toCurveX(0), curveCanvas.height);
      cCtx.stroke();

      for (let j = 0; j < HIDDEN_SIZE; j++) {
        cCtx.strokeStyle = 'rgba(33, 150, 243, 0.25)';
        cCtx.beginPath();
        for (let point = 0; point < curvePoints.length; point++) {
          const { px, h2 } = curvePoints[point];
          const py = toCurveY(h2[j] * w3[j]);
          if (point === 0) cCtx.moveTo(px, py);
          else cCtx.lineTo(px, py);
        }
        cCtx.stroke();
      }

      cCtx.strokeStyle = '#4caf50';
      cCtx.lineWidth = 3;
      cCtx.beginPath();
      for (let point = 0; point < curvePoints.length; point++) {
        const { px, x } = curvePoints[point];
        const py = toCurveY(bellCurve(x));
        if (point === 0) cCtx.moveTo(px, py);
        else cCtx.lineTo(px, py);
      }
      cCtx.stroke();

      cCtx.strokeStyle = '#ff5252';
      cCtx.lineWidth = 3;
      cCtx.beginPath();
      for (let point = 0; point < curvePoints.length; point++) {
        const { px, out } = curvePoints[point];
        const pred = out;
        const py = toCurveY(pred);
        if (point === 0) cCtx.moveTo(px, py);
        else cCtx.lineTo(px, py);
      }
      cCtx.stroke();

      epochDisplay.textContent = epoch.toLocaleString();
      trainingLossDisplay.textContent = trainingLoss === null
        ? '—'
        : trainingLoss.toFixed(6);
      evaluationLossDisplay.textContent = calculateEvaluationMSE().toFixed(6);
    }

    function drawActivationStats() {
      aCtx.clearRect(0, 0, actCanvas.width, actCanvas.height);

      const layerPreferences = [actLogits1, actLogits2].map(layerLogits => {
        const preferences = new Array(NUM_ACTS).fill(0);
        for (const neuronLogits of layerLogits) {
          const probabilities = softmax(neuronLogits);
          for (let k = 0; k < NUM_ACTS; k++) {
            preferences[k] += probabilities[k] / HIDDEN_SIZE;
          }
        }
        return preferences;
      });

      aCtx.fillStyle = '#e0e0e0';
      aCtx.font = 'bold 14px sans-serif';
      aCtx.textAlign = 'center';
      aCtx.fillText('Learned Activation Preferences', actCanvas.width / 2, 25);
      aCtx.font = '11px sans-serif';
      aCtx.fillStyle = '#aaa';
      aCtx.fillText('Mean softmax probability per layer', actCanvas.width / 2, 44);

      for (let layer = 0; layer < layerPreferences.length; layer++) {
        const sectionTop = 73 + layer * 152;
        aCtx.textAlign = 'left';
        aCtx.fillStyle = '#e0e0e0';
        aCtx.font = 'bold 13px sans-serif';
        aCtx.fillText(`Hidden layer ${layer + 1}`, 18, sectionTop);

        for (let k = 0; k < NUM_ACTS; k++) {
          const y = sectionTop + 18 + k * 29;
          const preference = layerPreferences[layer][k];
          const barX = 92;
          const barWidth = 146;

          aCtx.fillStyle = '#ccc';
          aCtx.font = '11px sans-serif';
          aCtx.fillText(CANDIDATES[k].name, 18, y + 12);

          aCtx.fillStyle = '#2a2a2a';
          aCtx.fillRect(barX, y, barWidth, 15);
          aCtx.fillStyle = CANDIDATES[k].color;
          aCtx.fillRect(barX, y, barWidth * preference, 15);

          aCtx.fillStyle = '#fff';
          aCtx.textAlign = 'right';
          aCtx.font = '11px monospace';
          aCtx.fillText(`${(preference * 100).toFixed(0)}%`, actCanvas.width - 14, y + 12);
        }
      }
    }

    function loop() {
      if (isRunning) {
        for (let i = 0; i < STEPS_PER_FRAME; i++) {
          trainStep();
        }
        drawCurves();
        drawActivationStats();
      }
      if (populationRunning) {
        advancePopulation();
      }
      requestAnimationFrame(loop);
    }

    toggleBtn.addEventListener('click', () => {
      isRunning = !isRunning;
      toggleBtn.textContent = isRunning ? 'Pause' : 'Resume';
    });

    resetBtn.addEventListener('click', () => {
      initNetwork();
      population = null;
      populationRunning = false;
      populationStep = 0;
      generation = 0;
      compareBtn.textContent = 'Start AI population';
      comparisonResults.textContent = '';
      comparisonStatus.textContent = 'Population is stopped.';
      drawCurves();
      drawActivationStats();
    });

    compareBtn.addEventListener('click', togglePopulation);

    initNetwork();
    drawCurves();
    drawActivationStats();
    loop();
