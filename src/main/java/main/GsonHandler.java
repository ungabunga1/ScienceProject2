package main;

import model.Set;
import com.google.gson.Gson;
import multiThreading.Threader;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class GsonHandler {

    public static Gson g = new Gson();

    public static Set[] loadSets() throws IOException {

        String[] names = new String[]{"training_1","training_2","training_3","training_4","training_5","testing"};

        System.out.print("LOADING DATASET > ");
        Set[] Sets = Threader.loadSets(names);
        System.out.println("DONE\n");

        return Sets;
    }

    public static Set loadSet(String name) {
        try {
            Set set = g.fromJson(IOUtils.resourceToString("/CIFAR10/"+name+".json", StandardCharsets.UTF_8), Set.class).init();
            set.init();
            return set;
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
        return null;
    }
}
