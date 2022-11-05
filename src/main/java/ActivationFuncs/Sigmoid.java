package ActivationFuncs;

public class Sigmoid implements IActivationFunctions{

    @Override
    public double Function(double x) {
        return 1f/(1f + Math.exp(-x)); // math.exp(x) is e^x || Euler's number raised to the power of x
    }

    @Override
    public double Derivative(double x) {
        return Function(x)*(1f - Function(x));
    }
}
