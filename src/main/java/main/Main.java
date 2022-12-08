package main;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        System.out.println("hello world \n\n\n");

        // change this to 2 when testing second group
        int group = 2;

        Experiment experiment = new Experiment(group,100, 5); // epochs are how many times the neural network will be fed a training set before testing
        experiment.init();
        experiment.run();


        long end = System.currentTimeMillis();
        System.out.println("Finished in "+(end-start)+" milliseconds... yikes");
    }



}
