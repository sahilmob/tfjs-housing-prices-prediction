import * as tf from "@tensorflow/tfjs";

const INPUTS = [];
const OUTPUTS = [];

const LEARNING_RATE = 0.0001;
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

function normalize(tensor: tf.Tensor, min?: tf.Tensor, max?: tf.Tensor) {
  const result = tf.tidy(() => {
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
}

for (let n = 1; n <= 20; n++) {
  INPUTS.push(n);
}

for (let n = 0; n < INPUTS.length; n++) {
  OUTPUTS.push(INPUTS[n] * INPUTS[n]);
}

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor1d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

const FEATURE_RESULT = normalize(INPUTS_TENSOR);

INPUTS_TENSOR.dispose();

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [1], units: 25, activation: "relu" }));
model.add(tf.layers.dense({ units: 5, activation: "relu" }));
model.add(tf.layers.dense({ units: 1 }));

model.summary();

function evaluate() {
  tf.tidy(function () {
    const newInput = normalize(
      tf.tensor1d([7]),
      FEATURE_RESULT.MIN_VALUES,
      FEATURE_RESULT.MAX_VALUES
    );

    const output = model.predict(newInput.NORMALIZED_VALUES);
    console.log(output.toString());

    FEATURE_RESULT.MIN_VALUES.dispose();
    FEATURE_RESULT.MAX_VALUES.dispose();
    model.dispose();

    console.log(tf.memory().numTensors);
  });
}

function logProgress(epoch: number, logs?: tf.Logs) {
  if (logs) console.log("Data for epoch " + epoch, Math.sqrt(logs.loss));

  if (epoch === 70) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  }
}

async function train() {
  model.compile({
    optimizer: OPTIMIZER,
    loss: "meanSquaredError",
  });

  const result = await model.fit(
    FEATURE_RESULT.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      callbacks: { onEpochEnd: logProgress },
      shuffle: true,
      batchSize: 2,
      epochs: 200,
    }
  );

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULT.NORMALIZED_VALUES.dispose();

  console.log(
    "Avg error loss: " +
      Math.sqrt(result.history.loss[result.history.loss.length - 1] as number)
  );

  evaluate();
}

train();

function App() {
  return <>hello world</>;
}

export default App;
