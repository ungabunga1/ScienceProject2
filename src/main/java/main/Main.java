package main;

import linearalgebra.Vector;
import model.NeuralNet;
import model.Set;

import java.io.IOException;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws IOException {
        System.out.println("welcome Layers, Neural Network, linearalgebra, and forward propagation! :) \n\n\n");

        //Set.initSets();

        NeuralNet nn = new NeuralNet(1,1, new int[]{1});


        System.out.println("output: "+nn.parse(new Vector(5)));
        System.out.println(nn);
    }



}
