import * as tf from "@tensorflow/tfjs";
import { TRAINING_DATA } from "./assets/real-estate-data";

const INPUTS = TRAINING_DATA.inputs;
const OUTUPTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTUPTS);

function App() {
  return <>hello world</>;
}

export default App;
