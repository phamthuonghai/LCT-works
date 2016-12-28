import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        mainProd("../English.txt", "../English_prefix.out", true);
        mainProd("../English.txt", "../English_suffix.out", false);
    }

    /**
     * myReverse
     * @param st input string
     * @return the reversed string
     */
    private static String myReverse(String st) {
        StringBuilder tmp = new StringBuilder(st);
        return tmp.reverse().toString();
    }

    /**
     * mainProd which does the affix discovery
     * @param inputFileName string
     * @param outputFileName string
     * @param prefix discover the prefix (if true) or suffix (false)
     */
    private static void mainProd(String inputFileName, String outputFileName, boolean prefix) {
        TrieDictionary myTrie = new TrieDictionary();
        Map<String, Integer> affixes = new HashMap<>();
        try {
            Instant start = Instant.now();
            Set<String> words = myTrie.fromFile(new File(inputFileName), prefix);
            Instant end = Instant.now();
            System.out.println("Constructing tree from file: " + Duration.between(start, end));

            start = Instant.now();
            for (String w : words) {
                myTrie.findAffixes(w, affixes, false);
            }
            end = Instant.now();
            System.out.println("Discovering affixes: " + Duration.between(start, end));
        } catch (IOException e) {
            e.printStackTrace();
        }

        try(  PrintWriter out = new PrintWriter(outputFileName)  ){
            affixes.entrySet().stream()
                    .filter(w -> w.getValue() > 0)                                  // filter all non-positive score
                    .sorted((k1, k2) -> -k1.getValue().compareTo(k2.getValue()))    // sort affixes based on score
                    .forEach(k -> out.println((prefix ? myReverse(k.getKey()) : k.getKey()) + " " + k.getValue()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * this is purely for testing purpose (instead of writing unit test)
     */
    private static void mainTest1() {
        TrieDictionary myTrie = new TrieDictionary();
        Map<String, Integer> goodWords = new HashMap<>();
        ArrayList<String> badWords = new ArrayList<>();

        goodWords.put("report", 3900);
        goodWords.put("reporter", 241);
        goodWords.put("reporters", 82);
        goodWords.put("reported", 609);
        goodWords.put("reportable", 5);
        goodWords.put("reportage", 63);

        badWords.add("repo");
        badWords.add("save"); //not a bad word, but for now not part of dictionary
        badWords.add("t");
        badWords.add("ejis");
        badWords.add("repe");

        System.out.println("−−−−−−− Adding Words to Trie −−−−−−−");
        for (Map.Entry<String, Integer> entry : goodWords.entrySet())
        {
            System.out.println("Adding " + entry.getKey() + " " + myTrie.AddWord(entry.getKey(), entry.getValue()));
        }
        System.out.println("−−−−−−− Checking if all good words return true −−−−−−−");
        for (Map.Entry<String, Integer> entry : goodWords.entrySet())
        {
            System.out.println("Checking for " + entry.getKey() + " " + myTrie.hasEntry(entry.getKey()));
        }
        System.out.println("−−−−−−− Checking that all list of bad words return false −−−−−−−");
        for (String s3 : badWords) {
            System.out.println("Checking for " + s3 + " " + myTrie.hasEntry(s3));
        }

        System.out.println("−−−−−−− Morphemes found from reporters −−−−−−−");
        Map<String, Integer> morphemes = new HashMap<>();
        myTrie.findAffixes("reporters", morphemes, true);
        System.out.println(morphemes);
    }
}
