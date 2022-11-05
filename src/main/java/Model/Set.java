package Model;

public class Set {
    int[][] data;
    int[] labels;
    String[] labelNames;

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

    public String[] getLabelNames() {
        return labelNames;
    }

    public void setLabelNames(String[] labelNames) {
        this.labelNames = labelNames;
    }
}
