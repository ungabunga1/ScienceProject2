package model.layers;

import activationFuncs.LeakyRelu;
import linearalgebra.Matrix;
import linearalgebra.Vector;

public class HiddenLayer extends Layer {
    public HiddenLayer(int numberOfNeurons, int numberOfInputs) {
        this.numberOfNeurons = numberOfNeurons;
        this.activationFunc = new LeakyRelu();
        this.inputs = new Vector(new double[numberOfInputs]);
        this.activations = new Vector(new double[numberOfNeurons]);

        this.weights = new Matrix(new double[numberOfNeurons][numberOfInputs]);
        this.biases = new Vector(new double[numberOfNeurons]);
    }
}
