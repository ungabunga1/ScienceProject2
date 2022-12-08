package main;

import model.NeuralNet;
import model.Set;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class Experiment {
    int group;
    int trainingEpochs; // how many times back prop will be preformed on each training set
    int runs; // how many times it should be repeated

    private static final int inputs = 3072;
    private static final int outputs = 10;
    private static final int[] hiddenNeurons = new int[] {2,2};
    private static final int splits = 100;
    private static final int maxGradientArraySize = 2; // determines how many gradients can be created before joining into 1
    private static final double learningRate = 0.0001;

    public Experiment(int group, int trainingEpochs, int runs) {
        if (group != 1 && group != 2) {
            throw new IllegalArgumentException();
        }
        this.group = group;
        this.trainingEpochs = trainingEpochs;
        this.runs = runs;
    }

    public void init() {
        try {
            Set.initSets();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void run() {
        for (int i = 1; i < runs+1; i++) {
            if (group == 1) { // normal sets
                run(Arrays.copyOfRange(Set.Sets, 0, Set.Sets.length - 1), Set.Sets[Set.Sets.length - 1], "Group_1(non-random)_"+i);
            } else { // random sets
                run(Arrays.copyOfRange(Set.RandomSets, 0, Set.RandomSets.length - 1), Set.RandomSets[Set.RandomSets.length - 1], "Group_2(random)_"+i);
            }
        }
    }

    private void run(Set[] training, Set test, String testname) {
        NeuralNet nn = new NeuralNet(inputs,outputs,hiddenNeurons);


        StringBuilder results = new StringBuilder("------< " + testname + " Results >------\n");
        StringBuilder Costs = new StringBuilder("--<Costs for training epochs ");
        StringBuilder Accuracies = new StringBuilder("--<Accuracies for training epochs ");
/*        for (int i = 0; i < training.length; i ++) {
            Set set = training[i];
            results.append("< "+"Training Set "+i+" >\n");

            for (int o = 0; o < trainingEpochs; o++) {
                System.out.print("<"+o+"> ");
                boolean computeCost;
                computeCost = o == 0 || o == trainingEpochs-1 || o % (trainingEpochs/10f) < 1;
                double[] doubles = nn.backparse(set.getVectorData(),set.getDesired(),computeCost,computeCost,splits,maxGradientArraySize);
                if (computeCost) {
                    results.append("Cost: ").append(doubles[0]).append("\n");
                    results.append("Accuracy: ").append(doubles[1]).append("\n");
                }
                System.out.print("DONE \n");
            }
        }*/

        for (int o = 1; o < trainingEpochs+1; o++) {
            for (int i = 0; i < training.length; i++) {
                boolean compute =  (o == 1 && i == 0) || (o+1 >= trainingEpochs+1 && i+1 >= training.length) || (!(i+1 < training.length) && o % 5 == 0);
                if (compute) {

                    if (!(o+1 >= trainingEpochs+1 && i+1 >= training.length)) {
                        Costs.append(o);
                        Accuracies.append(o);
                        Costs.append(", ");
                        Accuracies.append(", ");
                    } else {
                        Costs.append("and ").append(o);
                        Accuracies.append("and ").append(o);
                    }
                }




            }
        }
        Costs.append(" respectively>--\n");
        Accuracies.append(" respectively>--\n");


        for (int o = 1; o < trainingEpochs+1; o++) {
            System.out.print(o+" > ");
            for (int i = 0; i < training.length; i++) {
                Set set = training[i];
                boolean compute =  (o == 1 && i == 0) || (o+1 >= trainingEpochs+1 && i+1 >= training.length) || (!(i+1 < training.length) && o % 5 == 0);

                double[] doubles = nn.backparse(set.getVectorData(),set.getDesired(),compute,compute,splits,maxGradientArraySize,learningRate);
                if (compute) {
                    System.out.print("|");
                } else {
                    System.out.print(":");
                }
                if (compute) {
                    Costs.append(doubles[0]).append("\n");
                    Accuracies.append(doubles[1]).append("\n");
                }

            }
            System.out.print("\n");
        }

        results.append(Costs).append("\n");
        results.append(Accuracies).append("\n");

        // test
        double[] doubles = nn.computeParse(test.getVectorData(),test.getDesired());
        results.append("--<Test>--").append("\n");
        results.append("Cost: ").append(doubles[0]).append("\n");
        results.append("Accuracy: ").append(doubles[1]).append("\n");

        File file = new File("Results/"+testname+".txt");
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter(file));
            bw.append(results);
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(results);
    }
}
