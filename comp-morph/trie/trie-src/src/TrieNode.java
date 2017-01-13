/**
 * Created by phamthuonghai on 11/25/16.
 */

import java.util.ArrayList;
import java.util.List;

public class TrieNode {
    private char nodeChar;
    private boolean endToken = false;
    private int tokenCount = 0;
    ArrayList<TrieNode> ChildNodes = new ArrayList<>();

    public TrieNode() { } //empty constructor
    public TrieNode(char c) {
        this.nodeChar = c;
    }

    /**
     * getNodeChar
     * @return nodeChar attribute
     */
    private char getNodeChar() {
        return this.nodeChar;
    }

    public TrieNode getChildNode(char c) {
        // iterate throughout all child nodes
        for (TrieNode node: ChildNodes) {
            if (node.getNodeChar() == c) {
                return node;
            }
        }

        return null;
    }

    /**
     * getAllChildNodes
     * @return all elements in ChildNodes attribute (wrapper in case ChildNodes is a map)
     */
    public List<TrieNode> getAllChildNodes() { return this.ChildNodes; }

    /**
     * addEntry adds the input string to Trie from this current node, created new child node if needed
     * @param input string
     * @param count occurrence of input string
     * @return true if succeeded
     */
    boolean addEntry(String input, int count) {
        if (input == null) {
            return false;
        }

        // increase token count for each node on the fly
        tokenCount += count;

        if (input.isEmpty()) {
            // end of word
            endToken = true;
            return true;
        }

        TrieNode t = this.getChildNode(input.charAt(0));

        // create new child node if not exist
        if (t == null) {
            t = new TrieNode(input.charAt(0));
            addChildNode(input.charAt(0), t);
        }

        return t.addEntry(input.substring(1), count);
    }

    /**
     * addChildNode add new node to be child of this current node, to be overwritten
     *              in other ChildNodes structure
     * @param c child's nodeChar
     * @param t the new (already initiated) node
     */
    public void addChildNode(Character c, TrieNode t) {
        this.ChildNodes.add(t);
    }

    /**
     * hasEntry checks if this word exists in the branch starting from this node
     * @param input input string
     * @return true if word exists in this TrieDictionary
     */
    boolean hasEntry(String input) {
        if (input == null) {
            return false;
        }

        if (input.isEmpty()) {
            return endToken;
        }

        TrieNode t = this.getChildNode(input.charAt(0));

        return t != null && t.hasEntry(input.substring(1));
    }

    /**
     * getTokenCount gets token count of the input string
     *                  return 0 if the input string does not exist in the branch
     *                  starting from this node
     * @param input input string
     * @return token count
     */
    int getTokenCount(String input)
    {
        if (input == null) {
            return 0;
        }

        if (input.isEmpty()) {
            return tokenCount;
        }

        TrieNode t = this.getChildNode(input.charAt(0));
        if (t == null) {
            return 0;
        } else {
            return t.getTokenCount(input.substring(1));
        }
    }
}

