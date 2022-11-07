package model;

import main.Randomizer;
import model.layers.HiddenLayer;
import model.layers.InputLayer;
import model.layers.Layer;
import linearalgebra.Matrix;
import linearalgebra.Vector;
import model.layers.OutputLayer;

import java.util.Arrays;

public class NeuralNet {
    Layer[] layers;

    Matrix[] weights;
    Vector[] biases;
    Vector[] activations;

    int[] numOfNeurons;
    int numOfInputs;
    int numOfOutputs;
    int numOfLayers;

    public NeuralNet(int numOfInputs, int numOfOutputs, int[] numOfHiddenNeurons) {
        this.numOfInputs = numOfInputs;
        this.numOfOutputs = numOfOutputs;
        this.numOfLayers = numOfHiddenNeurons.length+2;

        numOfNeurons = new int[numOfLayers];
        numOfNeurons[0] = numOfInputs;
        numOfNeurons[numOfNeurons.length-1] = numOfOutputs;

        System.arraycopy(numOfHiddenNeurons,0,numOfNeurons,1,numOfHiddenNeurons.length); // copies the hidden neurons array into the neurons array while leaving space for the input and output layers

        initWeightsAndBiases(-0.1f, 0.1f);
        initLayers();

        activations = new Vector[numOfLayers];
    }

    public Vector parse(Vector input) { // forward propagation :)
        activations[0] = layers[0].parse(input);
        for (int i = 1; i < numOfLayers; i++) {
            activations[i] = layers[i].parse(activations[i-1]);
        }
        return activations[activations.length-1];
    }

    private void initLayers() {
        layers = new Layer[numOfLayers];

        // create the layers
        for (int i = 0; i < layers.length; i++) {
            Layer layer;

            if (i == 0) {
                // if it's an input layer...
                layer = new InputLayer(numOfInputs);
            } else if (i == layers.length-1) {
                // if it's an output layer...
                layer = new OutputLayer(numOfNeurons[i],numOfNeurons[i-1]);
            } else {
                // if it's a hidden layer...
                layer = new HiddenLayer(numOfNeurons[i],numOfNeurons[i-1]);
            }

            layer.updateWeightsAndBiases(weights[i],biases[i]);
            layers[i] = layer;
        }
    }

    private void initWeightsAndBiases(float min, float max) {
        // min and max should be smaller numbers such as -0.1f and 0.1f
        populateWeights(min,max);
        populateBiases(min,max);
    }

    private void populateWeights(float min, float max) {
        weights = new Matrix[numOfLayers];

        weights[0] = null; // no weights for the input (first) layer

        // the loop skips the input layer since it's already done
        for (int i = 1; i < numOfLayers; i++) {
            weights[i] = new Matrix(Randomizer.randomDoubleArray(numOfNeurons[i],numOfNeurons[i-1],min,max));
        }
    }

    private void populateBiases(float min, float max) {
        biases = new Vector[numOfLayers];

        biases[0] = null; // no bias for input layer

        // the loop skips the input layer since it's already handled
        for(int i = 1; i < numOfLayers; i++) {
            biases[i] = new Vector(Randomizer.randomDoubleArray(numOfNeurons[i],min,max));
        }
    }

    public void updateLayers() throws Exception {
        for (int i = 0; i < numOfLayers; i++) {
            layers[i].updateWeightsAndBiases(weights[i],biases[i]);
        }
    }

    public String toString() {
        return "<NEURAL NETWORK>\n\n" +
                "Neurons: " +Arrays.toString(numOfNeurons)+
                "\nLayers: " +numOfLayers+
                "\nInputs: " +numOfInputs+
                "\nOutputs: " +numOfOutputs+

                "\n\nWeights:\n " +Arrays.toString(weights)+
                "\n\nBiases:\n "+Arrays.toString(biases)+
                "\n\nActivations:\n"+Arrays.toString(activations)+
                "\n\n</NEURAL NETWORK>\n";
    }

    // getters and setters

}
