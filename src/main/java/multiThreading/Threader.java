package multiThreading;

import libraries.linearalgebra.Matrix;
import libraries.linearalgebra.Vector;
import main.GsonHandler;
import model.NeuralNet;
import model.Set;
import model.backprop.Gradient;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class Threader {
    public static Set[] loadSets(String[] names) {
        return Arrays.stream(names).parallel().map(GsonHandler::loadSet).toArray(Set[]::new);
    }

    public static double[] computeCost(Vector[] outputs, Vector[] desiredOutputs) {
        List<Integer> all = IntStream.range(0,outputs.length).boxed().toList();
        return all.stream().parallel().map(i->NeuralNet.computeCost(outputs[i],desiredOutputs[i])).mapToDouble(i->i).toArray();
    }
    public static Vector[][] forward(Vector[] inputs, NeuralNet nn) {
        return Arrays.stream(inputs).parallel().map(nn::parse).toArray(Vector[][]::new); // black magic
    }


    public static Gradient backward(Vector[][] activations, Vector[] outputs, int maxGradientArraySize, double learningRate, NeuralNet nn) {
        int total = activations.length;
        if (total % maxGradientArraySize != 0) {
            System.out.println(total + " is not divisible by "+maxGradientArraySize);
            throw new IllegalArgumentException();
        }

        List<Vector[]> tempActs = new ArrayList<>(Arrays.stream(activations).toList());
        List<Vector> tempOuts = new ArrayList<>(Arrays.stream(outputs).toList());
        List<Gradient> tempGrads = new ArrayList<Gradient>();
        while (tempActs.size() > 0) {
            while (tempGrads.size() < maxGradientArraySize && tempActs.size() > 0) {
                Gradient grad = nn.computeGradient(tempActs.get(0),tempOuts.get(0),total,learningRate);
                tempGrads.add(grad);
                tempActs.remove(0);
                tempOuts.remove(0);
                //System.out.println("<");
            }
            Gradient grad = new Gradient(tempGrads.toArray(Gradient[]::new));
            tempGrads.clear();
            tempGrads.add(grad);
            //System.out.println("<<");
        }
        //System.out.println("<<<<<<<<<");

        if (tempGrads.size() != 1) {
            System.out.println("something went wrong here");
        }
        return tempGrads.get(0);

        // takes the product of forward and makes it go backward. Speech 100
        /*Vector[][][] splitActivations = Set.split(activations,maxGradientArraySize);
        Vector[][] splitOutputs = Set.split(outputs,maxGradientArraySize);
        List<Integer> all = IntStream.range(0,splitActivations.length).boxed().toList();
        Gradient[] gradients = all.stream().parallel().map(i->backward(splitActivations[i],splitOutputs[i],nn,total,learningRate)).toArray(Gradient[]::new);
        return new Gradient(gradients);*/
    }
    public static Gradient backward(Vector[][] activations, Vector[] outputs, NeuralNet nn,int total,double learningRate) {
        List<Integer> all = IntStream.range(0,activations.length).boxed().toList();

        Gradient[] gradients = all.stream().parallel().map(i->nn.computeGradient(activations[i],outputs[i],total,learningRate)).toArray(Gradient[]::new);
        return new Gradient(gradients);
    }

    public static Matrix weightSensitivities(Vector lastActivations, Vector zproduct, Vector sensitivity,int totalGradients,double learningRate, NeuralNet nn) {

        int rows = sensitivity.length();
        int columns = lastActivations.length();


        List<Integer> all = IntStream.range(0, rows * columns).boxed().toList();

        // hey, listen, i never said that i'm a good programmer okay?
        // i can just think outside the box... and come up with horrible solutions like this one
        // y x x
        // column = i - columns * (i / columns)
        // row = i / columns
        // there definitely has to be an easier way
        List<Double> doubles = all.stream().map(i->nn.getWeightSensitivity(lastActivations.get(i - columns * (i / columns)),sensitivity.get(i / columns),zproduct.get(i / columns),totalGradients,learningRate)).toList();

        return new Matrix(doubles.stream().mapToDouble(i->i).toArray(), rows); // made a new constructor for matrix
    }

    public static Vector biasSensitivities(Vector zproduct, Vector sensitivity, int totalGradients,double learningRate, NeuralNet nn) {
        List<Integer> all = IntStream.range(0,zproduct.length()).boxed().toList();
        List<Double> doubles =  all.stream().map(i->nn.getBiasSensitivity(sensitivity.get(i),zproduct.get(i),totalGradients,learningRate)).toList();
        return new Vector(doubles.stream().mapToDouble(i->i).toArray());
    }

    public static Vector activatonSensitivities(Matrix weights, Vector zproduct, Vector sensitivity, double totalGradients,double learningRate, NeuralNet nn) {
        List<Integer> all = IntStream.range(0,weights.getNumColumns()).boxed().toList();
        List<Double> doubles = all.stream().map(i->nn.getActivationSensitivity(weights.getColumn(i),sensitivity,zproduct,totalGradients,learningRate)).toList();
        return new Vector(doubles.stream().mapToDouble(i->i).toArray());
    }

    public static void greatestCommonFactor(int n1, int n2) {

    }
}
