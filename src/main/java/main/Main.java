package main;

import libraries.linearalgebra.Matrix;
import libraries.linearalgebra.Vector;
import model.NeuralNet;
import model.Set;
import model.backprop.Gradient;
import multiThreading.Threader;

import java.io.IOException;
import java.util.stream.IntStream;

public class Main {

    public static void main(String[] args) throws IOException {
        long start = System.currentTimeMillis();
        System.out.println("welcome multithreading! :) \n\n\n");

        //Set.initSets();

        NeuralNet nn = new NeuralNet(2,5, new int[]{10});

        Vector[] inputs = new Vector[]{new Vector(1,2)};
        Vector[] outputs = new Vector[]{new Vector(1,0,0,0,0)};

        for (int i = 1000; i > 0; i--) {
            System.out.println(nn.backparse(inputs,outputs,true));
        }


        long end = System.currentTimeMillis();
        System.out.println("Time : "+(end-start)+"ms");
    }



}
