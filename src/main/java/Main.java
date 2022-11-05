import java.io.IOException;
import java.util.Random;

public class Main {

    static Random rand = new Random(System.currentTimeMillis());

    public static void main(String[] args) throws IOException {
        System.out.println("welcome Gson! :) \n\n\n");

        GsonHandler.CreateSets();
    }

}
