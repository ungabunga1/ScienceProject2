package model;

import main.Randomizer;
import model.backprop.Gradient;
import model.layers.HiddenLayer;
import model.layers.InputLayer;
import model.layers.Layer;
import libraries.linearalgebra.Matrix;
import libraries.linearalgebra.Vector;
import model.layers.OutputLayer;
import multiThreading.Threader;

import java.util.Arrays;

public class NeuralNet {
    Layer[] layers;

    Matrix[] weights;
    Vector[] biases;

    final int[] numOfNeurons;
    final int numOfInputs;
    final int numOfOutputs;
    final int numOfLayers;

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


    }

    public Vector[] parse(Vector input) { // forward propagation :)
        Vector[] activations = new Vector[numOfLayers];// create a vector array to store the activations for each layer

        activations[0] = layers[0].parse(input); // setup the activation for the input layer

        for (int i = 1; i < numOfLayers; i++) { // iterate through the remaining layers, using the output of the previous layer
            activations[i] = layers[i].parse(activations[i-1]);
        }
        return activations;
    }

    public Vector[][] parse(Vector[] inputs) { // forward propagation but faster... maybe
        return Threader.forward(inputs,this);
    }

    public double backparse(Vector[] inputs, Vector[] outputs, boolean computeCost) { // back propagation but faster... maybe
        Vector[][] activations = parse(inputs);
        Gradient gradient = Threader.backward(activations,outputs,this);
        applyGradient(gradient);


        if (computeCost) {
            Vector[] neuralOutputs = new Vector[inputs.length];
            for (int i = 0; i < neuralOutputs.length; i++) {
                neuralOutputs[i] = activations[i][numOfLayers-1];
            }
            return computeCost(neuralOutputs,outputs);
        }
        return -99999.69;
    }

    public static double computeCost(Vector[] outputs, Vector[] desiredOutputs) {
        double[] doubles = Threader.computeCost(outputs,desiredOutputs);
        double averageCost = 0;
        for (int i = 0; i < doubles.length; i++) {
            averageCost += doubles[i] / doubles.length;
        }
        return averageCost;
    }

    public static double computeCost(Vector output, Vector desiredOutput) {
        double cost = 0;
        for (int i = 0; i < output.length(); i++) {
            cost += Math.pow(output.get(i) - desiredOutput.get(i),2);
        }
        return cost;
    }
    public void applyGradient(Gradient gradient) {
        Matrix[] weights = this.weights.clone();
        Vector[] biases = this.biases.clone();

        for (int i = 1; i < weights.length; i++) {
            weights[i] = weights[i].add(gradient.getWg()[i]);

        }
        for (int i = 1; i < biases.length; i++) {
            biases[i] = biases[i].add(gradient.getBg()[i]);
        }
        updateWeightAndBias(weights,biases);
    }

    public void updateWeightAndBias(Matrix[] weights, Vector[] biases) {
        this.weights = weights;
        this.biases = biases;

        try {
            updateLayers();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public Gradient computeGradient(Vector[] activations, Vector desiredOutput, int totalGradients) { // takes in vector input, spits out a gradient for that input
        Matrix[] ws = new Matrix[weights.length];
        Vector[] bs = new Vector[biases.length];

        // for input layer
        ws[0] = null;
        bs[0] = null;

        Vector actSens = activations[activations.length-1].subtract(desiredOutput).multiply(2); // get the activation sensitivity for the first ones
        for (int i = 0; i < numOfLayers-1; i ++) {
            int layerindex = (numOfLayers-1)-i;
            var sensitivities = calcSens(activations,actSens,i,totalGradients);
            ws[layerindex] = (Matrix) sensitivities[1];
            bs[layerindex] = (Vector) sensitivities[2];

            actSens = (Vector) sensitivities[0];
        }
        return new Gradient(ws,bs);
    }

    public Object[] calcSens(Vector[] activations, Vector activationSensitivity, int depth, int totalGradients){
        // get the relevant layer
        int layerindex = (numOfLayers-1)-depth;
        Layer layer = layers[layerindex];

        // get some variables from the layer
        Matrix lws = layer.getWeights(); // layer weights
        Vector lbs = layer.getBiases(); // layer biases

        // define some variables

        Vector z = lws.multiply(activations[layerindex-1]).add(lbs);
        Vector zproduct = layer.getActivationFunc().derivative(z);

        // calculate the weight sensitivity
        Matrix ws = Threader.weightSensitivities(activations[layerindex-1],zproduct,activationSensitivity,totalGradients,this);

        // calculate the bias sensitivity
        Vector bs = Threader.biasSensitivities(zproduct,activationSensitivity,totalGradients,this);

        // calculate the next activation sensitivity
        Vector as = Threader.activatonSensitivities(lws,zproduct,activationSensitivity,totalGradients,this);


        return new Object[]{as,ws,bs};
    }

    public double getWeightSensitivity(double lastActivation, double activationSensitivity, double zproduct, double totalgradients) {
        return (lastActivation * activationSensitivity * zproduct)/totalgradients;
    }

    public double getBiasSensitivity(double activationSensitivity, double zproduct, double totalgradients) {
        return (activationSensitivity*zproduct)/totalgradients;
    }

    public double getActivationSensitivity(Vector weights, Vector activationSensitivites, Vector zproduct, double totalgradients) {
        double sensitivity = 0;
        for (int i = 0; i < weights.length(); i++) {
            sensitivity += (weights.get(i)*activationSensitivites.get(i)*zproduct.get(i))/totalgradients;
        }
        return sensitivity;
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
                "\n\n</NEURAL NETWORK>\n";
    }

    // getters and setters

}
