package model.layers;

import libraries.linearalgebra.Matrix;
import libraries.linearalgebra.Vector;

public class InputLayer extends Layer {

    public InputLayer(int numberOfInputs) {
        // input layer has a lot of values null
        weights = null;
        biases = null;
        activationFunc = null;

        numberOfNeurons = numberOfInputs;

    }

    // the input layer is special
    @Override
    public Vector parse(Vector inputs) {
        return inputs;
    }
    @Override
    public void updateWeightsAndBiases(Matrix weights, Vector biases) {
        return;
    }
}
