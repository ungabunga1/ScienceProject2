package model.layers;

import activationFuncs.IActivationFunc;
import linearalgebra.Matrix;
import linearalgebra.Vector;

public abstract class Layer {
    Vector activations;
    Vector inputs;

    Matrix weights;
    Vector biases;

    int numberOfNeurons;
    IActivationFunc activationFunc;

    // forward propagation
    public Vector parse(Vector inputs) {
        inputs = new Vector(inputs);
        activations = activationFunc.Function(weights.multiply(inputs).add(biases)); // weight multiplied by the input, plus the bias, and through the activation function
        return activations;
    }

    // functions for updating the weights and biases
    public void updateWeightsAndBiases(Matrix weights, Vector biases) {
        try {
            updateWeights(weights);
            updateBiases(biases);
        } catch (IllegalArgumentException e) {
            System.out.println(e);
            System.out.println("Stopping The Show :(");
            System.exit(0);
        }
    }

    public void updateWeights(Matrix weights) throws IllegalArgumentException {
        // make sure the sizes are consistent
        if (this.weights != null && (this.weights.getNumColumns() != weights.getNumColumns() || this.weights.getNumRows() != weights.getNumRows())) {
            throw new IllegalArgumentException("Mismatching Matrix Dimensions During Weight Update");
        }

        this.weights = new Matrix(weights);
    }

    public void updateBiases(Vector biases) throws IllegalArgumentException {
        // make sure the sizes are consistent
        if (this.biases != null && this.biases.length() != biases.length()) {
            throw new IllegalArgumentException("Mismatching Vector Dimensions During Bias Update");
        }

        this.biases = new Vector(biases);
    }

    // getters and setters
    public Vector getActivations() {
        return activations;
    }

    public void setActivations(Vector activations) {
        this.activations = activations;
    }

    public Vector getInputs() {
        return inputs;
    }

    public void setInputs(Vector inputs) {
        this.inputs = inputs;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    public Vector getBiases() {
        return biases;
    }

    public void setBiases(Vector biases) {
        this.biases = biases;
    }

    public int getNumberOfNeurons() {
        return numberOfNeurons;
    }

    public void setNumberOfNeurons(int numberOfNeurons) {
        this.numberOfNeurons = numberOfNeurons;
    }

    public IActivationFunc getActivationFunc() {
        return activationFunc;
    }

    public void setActivationFunc(IActivationFunc activationFunc) {
        this.activationFunc = activationFunc;
    }
}
