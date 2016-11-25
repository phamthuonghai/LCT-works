import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        TrieDictionary myTrie = new TrieDictionary();
        ArrayList<String> goodWords = new ArrayList<String>();
        ArrayList<String> badWords = new ArrayList<String>();
        goodWords.add("report");
        goodWords.add("reporter");
        goodWords.add("reporters");
        goodWords.add("reported");
        goodWords.add("reportable");
        goodWords.add("reportage");
        goodWords.add("reportages");
        goodWords.add("reporting");
        badWords.add("repo");
        badWords.add("save");//not a bad word, but for now not part of dictionary
        badWords.add("t");
        badWords.add("ejis");
        badWords.add("repe");
        System.out.println("−−−−−−− Adding Words to Trie −−−−−−−");
        for (String s1 : goodWords) {
            System.out.println("Adding " + s1 + " " + myTrie.AddWord(s1));
        }
        System.out.println("−−−−−−− Checking if all good words return true −−−−−−−");
        for (String s2 : goodWords) {
            System.out.println("Checking for " + s2 + " " + myTrie.hasEntry(s2));
        }
        System.out.println("−−−−−−− Checking that all list of bad words return false −−−−−−−");
        for (String s3 : badWords) {
            System.out.println("Checking for " + s3 + " " + myTrie.hasEntry(s3));
        }
    }
}
