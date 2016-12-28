import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by phamthuonghai on 11/25/16.
 */
public class TrieDictionary {
    TrieNode rootNode;
    public TrieDictionary() {
        rootNode = new TrieNode();
    }

    /**
     * addWord adds a new word and its occurrence to this dictionary
     * @param word input word
     * @param count occurrence
     * @return true if succeeded
     */
    boolean AddWord(String word, int count) {
        return rootNode.addEntry(word, count);
    }

    /**
     * hasEntry checks if this TrieDictionary contains the input word
     * @param word input
     * @return true if word exists in this TrieDictionary
     */
    boolean hasEntry(String word) {
        return rootNode.hasEntry(word);
    }

    /**
     * fromFile reads all lines from a file object then construct our Trie
     * @param file input file, which is a file object
     * @param prefix for the function to decide reverse each string or not
     * @return a set of words in our database (i.e. the first entry in each line) for later use
     * @throws IOException when file not found or error during reading file
     */
    Set<String> fromFile(File file, boolean prefix) throws IOException {
        Scanner scanner = new Scanner(file);
        Set<String> stringSet = new HashSet<>();
        while(scanner.hasNext()){
            String[] tokens = scanner.nextLine().split(" ");

            if (prefix)
            {   // reverse string using StringBuilder as String does not provide this function
                StringBuilder tmp = new StringBuilder(tokens[0]);
                tokens[0] = tmp.reverse().toString();
            }

            if (tokens.length == 2) {   // check valid parsed tokens
                this.AddWord(tokens[0], Integer.parseInt(tokens[1]));
                stringSet.add(tokens[0]);
            }
        }
        return stringSet;
    }

    /**
     * getTokenCount
     * @param word the input string
     * @return the number of tokens start with 'word' (i.e. number of tokens have 'word' as prefix)
     */
    private int getTokenCount(String word)
    {
        if (word == null || word.isEmpty())
            return 0;
        else
            return rootNode.getTokenCount(word);
    }

    /**
     * findAffixes tries to cut the word down in 2 chunks and applies tests as described in our lecture
     * @param word input word
     * @param result a map String -> Integer, which includes an affix and its score
     * @param verbose print log or not
     */
    void findAffixes(String word, Map<String, Integer> result, boolean verbose)
    {
        for (int i = 1; i < word.length(); i++)
        {
            String alpha = word.substring(0, i-1);
            String alphaA = word.substring(0, i);
            String alphaAB = word.substring(0, i+1);
            String Bbeta = word.substring(i);
            if (verbose)
                System.out.println("\nBoundary check: " + alphaA + " - " + Bbeta);

            boolean test1 = this.hasEntry(alphaA);
            if (verbose) {
                System.out.println("Test 1: for " + alphaA + " is " + test1);

                System.out.println("Test 2: Freq of alphaA / Freq of alpha = is it approx equal to 1?");
            }

            int freqAlphaA = this.getTokenCount(alphaA);
            int freqAlpha = this.getTokenCount(alpha);
            boolean test2 = Math.abs((float) freqAlphaA / freqAlpha - 1) < 10e-3;
            if (verbose) {
                System.out.println("alphaA: " + freqAlphaA + " / alpha: " + freqAlpha + " = " + test2);

                System.out.println("Test 3: Freq of alphaAB / Freq of alphaA = is it much less than 1?");
            }

            int freqAlphaAB = this.getTokenCount(alphaAB);
            boolean test3 = (float) freqAlphaAB / freqAlphaA < 1;
            if (verbose)
                System.out.println("alphaAB: " + freqAlphaAB + " / alphaA: " + freqAlphaA + " = " + test3);

            int val = (test1 && test2 && test3) ? 19 : -1;

            Integer count = result.get(Bbeta);
            if (count == null) {
                result.put(Bbeta, val);
            }
            else {
                result.put(Bbeta, count + val);
            }
        }
    }

//    /**
//     * findAllMorphemes uses recursion to traverse through our Trie
//     * in order to improve running time by reducing check for each word
//     * @param result returns the affix and its score
//     */
//    void findAllAffixes(Map<String, Integer> result)
//    {
//        for (TrieNode node: rootNode.getAllChildNodes()) {
//            traverseTrie(node, rootNode, "", result);
//        }
//    }
//
//    /**
//     *
//     * @param node
//     * @param parent
//     * @param Bbeta
//     * @param result
//     */
//    private void traverseTrie(TrieNode node, TrieNode parent, String Bbeta, Map<String, Integer> result)
//    {
//        if (Bbeta.equals("")) {                         // This node is still in alphaA
//            if (node.endToken) {                            // Consider this node is A in alphaA, pass test 1
//                                                            // Prepare test 2
//                float freqAlphaA = node.tokenCount;
//                float freqAlpha = parent.tokenCount;
//                if (Math.abs(freqAlphaA / freqAlpha - 1) < 10e-3) { // Test 2 passed
//                    for (TrieNode child: node.getAllChildNodes()) {
//                        traverseTrie(child, node, String.valueOf(child.nodeChar), result);
//                    }
//                }
//            }
//
//            for (TrieNode child: node.getAllChildNodes()) {
//                traverseTrie(child, node, "", result);
//            }
//        }
//        else if (Bbeta.length() == 1){                  // This node is B
//                                                            // Prepare test 3
//            float freqAlphaAB = node.tokenCount;
//            float freqAlphaA = parent.tokenCount;
//            if (!(freqAlphaAB / freqAlphaA < 1)) {
//                return;                                     // Test failed
//            }
//            for (TrieNode child: node.getAllChildNodes()) {
//                traverseTrie(child, node, Bbeta + child.nodeChar, result);
//            }
//        }
//        else {                                          // This node is in beta
//            if (node.endToken) {                              // end of Bbeta
//        }
//    }
}
