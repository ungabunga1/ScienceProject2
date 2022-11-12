package activationFuncs;

public class Sigmoid implements IActivationFunc {

    private final float a = 1;

    @Override
    public double function(double x) {
        return 1f/(1f + Math.exp(-a*x)); // math.exp(x) is e^x || Euler's number raised to the power of x
    }

    @Override
    public double derivative(double x) {
        return function(x)*(1f - function(x));
    }
}
