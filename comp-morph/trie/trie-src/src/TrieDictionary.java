/**
 * Created by phamthuonghai on 11/25/16.
 */
public class TrieDictionary {
    TrieNode rootNode;
    public TrieDictionary() {
        rootNode = new TrieNode();
    }
    public boolean AddWord(String word, int count) {
        return rootNode.addEntry(word, count);
    }
    public boolean hasEntry(String word) {
        return rootNode.hasEntry(word);
    }

    public int getTokenCount(String word)
    {
        if (word == null || word.isEmpty())
            return 0;
        else
            return rootNode.getTokenCount(word);
    }
}
