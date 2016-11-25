/**
 * Created by phamthuonghai on 11/25/16.
 */
public class TrieDictionary {
    TrieNode rootNode;
    public TrieDictionary() {
        rootNode = new TrieNode();
    }
    public boolean AddWord(String word) {
        return rootNode.addEntry(word);
    }
    public boolean hasEntry(String word) {
        return rootNode.hasEntry(word);
    }
}
