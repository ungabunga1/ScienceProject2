package model.layers;

import activationFuncs.Sigmoid;
import linearalgebra.Matrix;
import linearalgebra.Vector;

public class OutputLayer extends Layer {
    public OutputLayer(int numberOfNeurons, int numberOfInputs) {
        this.numberOfNeurons = numberOfNeurons;
        this.activationFunc = new Sigmoid();
        this.inputs = new Vector(new double[numberOfInputs]);
        this.activations = new Vector(new double[numberOfNeurons]);

        this.weights = new Matrix(new double[numberOfNeurons][numberOfInputs]);
        this.biases = new Vector(new double[numberOfNeurons]);
    }
}
