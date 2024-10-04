import * as tf from "@tensorflow/tfjs";
import { TRAINING_DATA } from "./assets/real-estate-data";

function normalize(tensor: tf.Tensor, min: tf.Tensor, max: tf.Tensor) {
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

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

const FEATURE_RESULT = normalize(INPUTS_TENSOR);

INPUTS_TENSOR.dispose();

function App() {
  return <>hello world</>;
}

export default App;
