package activationFuncs;

public class LeakyRelu implements IActivationFunc {

    private final float mult = 0.1f;

    @Override
    public double function(double x) {
        if (x >= 0) {
            return x;
        } else {
            return x*mult;
        }
    }

    @Override
    public double derivative(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return mult;
        }
    }
}
