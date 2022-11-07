package model;


import main.GsonHandler;
import main.Randomizer;
import linearalgebra.Vector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Set {
    // variables
    public static Set[] Sets;
    public static Set[] RandomSets;

    protected int[][] data; // data of the dataset. each row is an image. each row is split into 3 1024 sections that represent the red, green, and blue values of an image respectively
    protected int[] labels; // labels for the data. the "correct answers"
    protected  Vector[] desired; // labels but in a vector form
    protected String[] labelNames; // what the labels actually mean

    public static void initSets() throws IOException {
        Sets = GsonHandler.LoadSets();
        RandomSets = randomizeSets(Sets);
    }


    public Set(int[][] data, int[] labels, String[] labelNames) {
        this.data = data;
        this.labels = labels;
        this.labelNames = labelNames;
        this.init();
    }

    public Set init() { // ran when generated by gson object to populate the desired class
        desired = new Vector[labels.length]; // every value automatically set to 0

        for (int i = 0; i < labels.length; i++) { // loop through every label and create an array of doubles that represent the desired neuron output
            desired[i] = new Vector(new double[labelNames.length]);
            desired[i] = desired[i].set(labels[i], 1.0f);
        }

        return this;
    }

    public static Set[] randomizeSets(Set[] sets) {
        Set[] randomSets = new Set[sets.length];

        for (int i = 0; i < randomSets.length; i++) {
            randomSets[i] = sets[i].randomizeSet();
        }

        return randomSets;
    }
    public Set randomizeSet() { // ony labels change

        List<Integer> randomLabels = new ArrayList<>(Arrays.stream(labels).boxed().toList());

        Collections.shuffle(randomLabels, Randomizer.rand);

        return new Set(data.clone(),randomLabels.stream().mapToInt(i->i).toArray(),labelNames.clone());
    }

    // getters and setters
    public int[][] getData() {
        return data;
    }

    public void setData(int[][] data) {
        this.data = data;
    }

    public int[] getLabels() {
        return labels;
    }

    public void setLabels(int[] labels) {
        this.labels = labels;
    }

    public Vector[] getDesired() {
        return desired;
    }

    public void setDesired(Vector[] desired) {
        this.desired = desired;
    }

    public String[] getLabelNames() {
        return labelNames;
    }

    public void setLabelNames(String[] labelNames) {
        this.labelNames = labelNames;
    }
}