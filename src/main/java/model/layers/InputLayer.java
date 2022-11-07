package model.layers;

import linearalgebra.Matrix;
import linearalgebra.Vector;

public class InputLayer extends Layer {

    public InputLayer(int numberOfInputs) {
        // input layer has a lot of values null
        weights = null;
        biases = null;
        activationFunc = null;

        numberOfNeurons = numberOfInputs;

        inputs = new Vector(new double[numberOfInputs]);
        activations = new Vector(new double[numberOfNeurons]);
    }

    // the input layer is special
    @Override
    public Vector parse(Vector inputs) {
        activations = new Vector(inputs);
        inputs = new Vector(inputs);
        return activations;
    }
    @Override
    public void updateWeightsAndBiases(Matrix weights, Vector biases) {
        return;
    }
}
