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
import java.util.List;
import java.util.stream.IntStream;

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

    public double[] computeParse(Vector[] inputs, Vector[] outputs) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException();
        }

        Vector[][] activations = parse(inputs);
        Vector[] neuralOutputs = new Vector[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            neuralOutputs[i] = activations[i][activations[i].length-1];
        }

        return new double[]{computeCost(neuralOutputs,outputs),computeAccuracy(neuralOutputs,outputs)};
    }

    public double[] backparse(Vector[] inputs, Vector[] outputs, boolean computeCost, boolean computeAccuracy, int splits, int maxGradientArraySize, double learningRate) {
        double cost = 0;
        double accuracy = 0;

        long seed = Randomizer.rand.nextLong();
        // using the same seed means that the numbers will be randomized the exact same way, which in this case, is good
        Vector[][] splitInputs = Set.randomlySplit(inputs,splits,seed);
        Vector[][] splitOutputs = Set.randomlySplit(outputs,splits,seed);

        List<Integer> all = IntStream.range(0,splits).boxed().toList();
        //List<List<Integer>> // have to split this up
        //add .parallel()
        double[][] doubles = all.stream().parallel().map(i->backparse(splitInputs[i],splitOutputs[i],maxGradientArraySize,learningRate,computeCost,computeAccuracy)).toArray(double[][]::new);

        if (computeCost || computeAccuracy) {
            for (int i = 0; i < splits; i++) {
                if (computeCost) {
                    cost += (double) doubles[i][0] / doubles.length;
                }
                if (computeAccuracy) {
                    accuracy += (double) doubles[i][1] / doubles.length;
                }
            }
        }

        return new double[]{cost,accuracy};
    }

    public double[] backparse(Vector[] inputs, Vector[] outputs, int maxGradientArraySize,double learningRate, boolean computeCost, boolean computeAccuracy) { // back propagation but faster... maybe
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException();
        }


        Vector[][] activations = parse(inputs);
        Gradient gradient = Threader.backward(activations,outputs,maxGradientArraySize,learningRate,this);
        applyGradient(gradient);


        Vector[] neuralOutputs = new Vector[inputs.length];
        if (computeCost || computeAccuracy) {
            for (int i = 0; i < neuralOutputs.length; i++) {
                neuralOutputs[i] = activations[i][numOfLayers - 1];
            }
        }

        double cost = -1;
        if (computeCost) {
            cost = computeCost(neuralOutputs,outputs);
            //System.out.println(cost);
        }

        double accuracy = -1;
        if (computeAccuracy) {
            accuracy = computeAccuracy(neuralOutputs,outputs);
        }

        return new double[]{cost, accuracy};
    }

    public static double computeAccuracy(Vector[] outputs, Vector[] desiredOutputs) {
        double total = outputs.length;
        double accuracy = 0;
        for (int i = 0; i < total; i++) {
            int mostActive = determineOutput(outputs[i]);
            int mostActiveDesried = determineOutput(desiredOutputs[i]);

            if (false) {
                System.out.println(outputs[i].toString());
                System.out.println(desiredOutputs[i].toString());
                System.out.println("----");
                System.out.println(mostActive);
                System.out.println(mostActiveDesried);
                System.out.println("? : " + (mostActive == mostActiveDesried));
                System.out.println("---------------------------");
            }

            if (mostActive == mostActiveDesried) {
                accuracy ++;
            } else{
                //System.out.println(mostActive +">"+outputs[i].get(mostActive)+" || "+ mostActiveDesried+ ">"+ desiredOutputs[i].get(mostActiveDesried));
            }
        }
       // System.out.println("why does it do this so many times?");
        //System.out.println(accuracy +" / "+total);
        return accuracy/total;
    }

    public static int determineOutput(Vector outputs) {
        int index = 0;
        for (int i = 0; i < outputs.length(); i++) {
            if (outputs.get(i) > outputs.get(index)){
                index = i;
            }
        }
        return index;
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
            weights[i] = weights[i].subtract(gradient.getWg()[i]);

        }
        for (int i = 1; i < biases.length; i++) {
            biases[i] = biases[i].subtract(gradient.getBg()[i]);
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


    public Gradient computeGradient(Vector[] activations, Vector desiredOutput, int totalGradients, double learningRate) { // takes in vector input, spits out a gradient for that input
        Matrix[] ws = new Matrix[weights.length];
        Vector[] bs = new Vector[biases.length];

        // for input layer
        ws[0] = null;
        bs[0] = null;

        Vector actSens = activations[activations.length-1].subtract(desiredOutput).multiply(2); // get the activation sensitivity for the first ones
        for (int i = 0; i < numOfLayers-1; i ++) {
            int layerindex = (numOfLayers-1)-i;
            var sensitivities = calcSens(activations,actSens,i,totalGradients,learningRate);
            ws[layerindex] = (Matrix) sensitivities[1];
            bs[layerindex] = (Vector) sensitivities[2];

            actSens = (Vector) sensitivities[0];
        }
        return new Gradient(ws,bs);
    }

    public Object[] calcSens(Vector[] activations, Vector activationSensitivity, int depth, int totalGradients, double learningRate){
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
        Matrix ws = Threader.weightSensitivities(activations[layerindex-1],zproduct,activationSensitivity,totalGradients,learningRate,this);

        // calculate the bias sensitivity
        Vector bs = Threader.biasSensitivities(zproduct,activationSensitivity,totalGradients,learningRate,this);

        // calculate the next activation sensitivity
        Vector as = Threader.activatonSensitivities(lws,zproduct,activationSensitivity,totalGradients,learningRate,this);


        return new Object[]{as,ws,bs};
    }

    public double getWeightSensitivity(double lastActivation, double activationSensitivity, double zproduct, double totalgradients, double learningRate) {
        return ((lastActivation * activationSensitivity * zproduct)/totalgradients)*learningRate;

    }

    public double getBiasSensitivity(double activationSensitivity, double zproduct, double totalgradients, double learningRate) {
        return ((activationSensitivity*zproduct)/totalgradients)*learningRate;
    }

    public double getActivationSensitivity(Vector weights, Vector activationSensitivites, Vector zproduct, double totalgradients, double learningRate) {
        double sensitivity = 0;
        for (int i = 0; i < weights.length(); i++) {
            sensitivity += ((weights.get(i)*activationSensitivites.get(i)*zproduct.get(i))/totalgradients)*learningRate;
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
