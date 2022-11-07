package main;

import java.util.Random;

public class Randomizer {
    public static Random rand = new Random(System.currentTimeMillis());

    public static float randomFloat(float min, float max) {
        return min + (max - min) * rand.nextFloat();
    }

    public static double[][] randomDoubleArray(int rows, int columns, float min, float max) {
        double[][] doubles = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            doubles[i] = randomDoubleArray(columns,min,max);
        }
        return doubles;
    }

    public static double[] randomDoubleArray(int length, float min, float max) {
        double[] doubles = new double[length];

        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = randomFloat(min, max);
        }

        return doubles;
    }
}
