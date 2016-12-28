/**
 * Created by phamthuonghai on 12/28/16.
 */

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrieNodeAdv extends TrieNode {
    Map<Character, TrieNode> ChildNodes = new HashMap<>();

    public TrieNode getChildNode(char c) {
        return ChildNodes.get(c);
    }

    public void addChildNode(Character c, TrieNode t) {
        this.ChildNodes.put(c, t);
    }

    public List<TrieNode> getAllChildNodes() { return (List)this.ChildNodes.values(); }
}
