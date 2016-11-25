/**
 * Created by phamthuonghai on 11/25/16.
 */
public class TrieNode {
    char nodeChar;
    boolean endToken = false;
    TrieNode[] ChildNodes = new TrieNode[0];
    public TrieNode() { } //empty constructor
    public TrieNode(char c) {
        this.nodeChar = c;
    }
    public char getNodeChar() {
        return this.nodeChar;
    }
    public TrieNode getChildNode(char c) {
        //search through ChildNodes[]
        //if you find a node represented by c
        //then return that node
        //else return null
        return null;
    }
    public boolean addEntry(String input) {
        //take the first char of the input string
        //and check whether it exists, and if not
        //create a childNode with that char
        //use the childNode to addEntry with
        //the remaining of the String
        //When you reach the last letter of the input string
        //set endToken = true
        //return true upon success
        return false;
    }
    public boolean hasEntry(String input) {
        return false;
    }
}
