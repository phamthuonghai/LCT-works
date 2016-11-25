/**
 * Created by phamthuonghai on 11/25/16.
 */

import java.util.ArrayList;

public class TrieNode {
    char nodeChar;
    boolean endToken = false;
    ArrayList<TrieNode> ChildNodes = new ArrayList<>();

    public TrieNode() { } //empty constructor
    public TrieNode(char c) {
        this.nodeChar = c;
    }

    public char getNodeChar() {
        return this.nodeChar;
    }

    public TrieNode getChildNode(char c) {
        for (TrieNode node: ChildNodes) {
            if (node.getNodeChar() == c) {
                return node;
            }
        }

        return null;
    }

    public boolean addEntry(String input) {
        if (input == null) {
            return false;
        }

        if (input.isEmpty()) {
            endToken = true;
            return true;
        }

        TrieNode t = this.getChildNode(input.charAt(0));
        if (t == null) {
            t = new TrieNode(input.charAt(0));
            this.ChildNodes.add(t);
        }

        return t.addEntry(input.substring(1));
    }

    public boolean hasEntry(String input) {
        if (input == null) {
            return false;
        }

        if (input.isEmpty()) {
            return endToken;
        }

        TrieNode t = this.getChildNode(input.charAt(0));

        return t != null && t.hasEntry(input.substring(1));
    }
}
