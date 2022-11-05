package ActivationFuncs;

public class LeakyRelu implements IActivationFunctions{

    private final float mult = 0.1f;

    @Override
    public double Function(double x) {
        if (x >= 0) {
            return x;
        } else {
            return x*mult;
        }
    }

    @Override
    public double Derivative(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return mult;
        }
    }
}
