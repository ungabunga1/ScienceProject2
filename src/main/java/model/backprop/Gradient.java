package model.backprop;

import libraries.linearalgebra.Matrix;
import libraries.linearalgebra.Vector;

public class Gradient { // container for a weight and bias gradient
    Matrix[] wg;
    Vector[] bg;

    public Gradient(Matrix[] wg, Vector[] bg) {
        this.wg = wg.clone();
        this.bg = bg.clone();
    }

    public Gradient(Gradient[] gradients) {
        wg = new Matrix[gradients[0].getWg().length];
        bg = new Vector[gradients[0].getBg().length];

        wg[0] = null;
        bg[0] = null;

        for (int i = 1; i < wg.length; i++) {
            Matrix m = new Matrix(gradients[0].getWg()[i]);
            for (int o = 1; o < gradients.length; o++) {
                m = m.add(gradients[o].getWg()[i]);
            }
            wg[i] = m;
        }

        for (int i = 1; i < bg.length; i++) {
            Vector v = new Vector(gradients[0].getBg()[i]);
            for (int o = 1; o < gradients.length; o++) {
                v = v.add(gradients[o].getBg()[i]);
            }
            bg[i] = v;
        }
    }

    // getters and setters

    public Matrix[] getWg() {
        return wg;
    }

    public void setWg(Matrix[] wg) {
        this.wg = wg;
    }

    public Vector[] getBg() {
        return bg;
    }

    public void setBg(Vector[] bg) {
        this.bg = bg;
    }
}
