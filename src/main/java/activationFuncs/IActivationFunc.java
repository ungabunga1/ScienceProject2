package activationFuncs;

import libraries.linearalgebra.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public interface IActivationFunc {

    default Vector function(Vector v){
        List<Double> list = new ArrayList<>(Arrays.stream(v.getEntries()).boxed().toList()); // turns the vector into an arraylist of Doubles
        double[] doubs = list.stream().map(this::function).toList().stream().mapToDouble(i -> i).toArray(); // runs each value in the array list through a function and turns it into an array of doubles

        return new Vector(doubs); // returns a vector made from the array of doubles
    }

    default Vector derivative(Vector v) {
        List<Double> list = new ArrayList<>(Arrays.stream(v.getEntries()).boxed().toList()); // turns the vector into a list
        double[] doubles = list.stream().map(this::derivative).mapToDouble(i->i).toArray(); // runs each double through derivative function
        return new Vector(doubles);
    }

    double function(double x);
    double derivative(double x);


}
