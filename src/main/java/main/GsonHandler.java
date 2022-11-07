package main;

import model.Set;
import com.google.gson.Gson;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class GsonHandler {

    public static Gson g = new Gson();

    public static Set[] LoadSets() throws IOException {
        Set[] Sets = new Set[6];

        System.out.print("LOADING DATASET > ");
        for (int i = 0; i < Sets.length; i++) {

            if (i == Sets.length-1) {
                Sets[i] = g.fromJson(IOUtils.resourceToString("/CIFAR10/testing.json", StandardCharsets.UTF_8), Set.class).init();
                continue;
            }

            Sets[i] = g.fromJson(IOUtils.resourceToString("/CIFAR10/training_"+(i+1)+".json", StandardCharsets.UTF_8), Set.class).init();
        }
        System.out.println("DONE\n");
        return Sets;
    }
}
