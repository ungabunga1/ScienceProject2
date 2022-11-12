package model.layers;

import activationFuncs.Sigmoid;
import libraries.linearalgebra.Matrix;
import libraries.linearalgebra.Vector;

public class OutputLayer extends Layer {
    public OutputLayer(int numberOfNeurons, int numberOfInputs) {
        this.numberOfNeurons = numberOfNeurons;
        this.activationFunc = new Sigmoid();

        this.weights = new Matrix(new double[numberOfNeurons][numberOfInputs]);
        this.biases = new Vector(new double[numberOfNeurons]);
    }
}
