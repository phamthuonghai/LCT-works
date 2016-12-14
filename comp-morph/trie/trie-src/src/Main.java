import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

public class Main {

    private static void testWord(TrieDictionary trie, String word)
    {
        for (int i = 1; i < word.length(); i++)
        {
            String alpha = word.substring(0, i-1);
            String alphaA = word.substring(0, i);
            String alphaAB = word.substring(0, i+1);
            String Bbeta = word.substring(i);
            System.out.println("\nBoundary check: " + alphaA + " - " + Bbeta);

            System.out.println("Test 1 for " + alphaA + " is " + trie.hasEntry(alphaA));

            System.out.println("Test 2: Freq of alphaA / Freq of alpha = is it approx equal to 1?");
            int freqAlphaA = trie.getTokenCount(alphaA);
            int freqAlpha = trie.getTokenCount(alpha);
            System.out.println("alphaA: " + freqAlphaA + " / alpha: " + freqAlpha + " = " +
                    (Math.abs((float)freqAlphaA/freqAlpha - 1) < 10e-3));

            System.out.println("Test 3: Freq of alphaAB / Freq of alphaA = is it much less than 1?");
            int freqAlphaAB = trie.getTokenCount(alphaAB);
            System.out.println("alphaAB: " + freqAlphaAB + " / alphaA: " + freqAlphaA + " = " +
                    ((float)freqAlphaAB/freqAlphaA < 1));
        }
    }

    public static void main(String[] args) {
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

        testWord(myTrie, "reporters");
    }
}
