package activationFuncs;

import linearalgebra.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public interface IActivationFunc {

    default Vector Function(Vector v){
        List<Double> list = new ArrayList<>(Arrays.stream(v.getEntries()).boxed().toList()); // turns the vector into an arraylist of Doubles
        double[] doubs = list.stream().map(this::Function).toList().stream().mapToDouble(i -> i).toArray(); // runs each value in the array list through a function and turns it into an array of doubles

        return new Vector(doubs); // returns a vector made from the array of doubles
    }
    double Function(double x);
    double Derivative(double x);


}
