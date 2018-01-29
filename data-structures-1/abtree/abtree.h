#ifndef ABTREE_ABTREE_H
#define ABTREE_ABTREE_H

#define MAX 4

struct abnode
{
    int count;
    int keys[MAX+1];
    abnode *child[MAX+1];
} ;


class ABTree {
    private:
        int A, B;
        abnode* root;
        bool insert_node(int, abnode *, int *, abnode **, int *, int *);
        bool search(int, abnode *, int *);
        void set_key(int, abnode *, abnode *, int);
        void split(abnode *, int, int *, abnode **);

        bool del_node(int, abnode *, int *, int *);
        void clear_key(abnode *, int);
        void copy_succ(abnode *, int);
        void restructure(abnode *, int);
        void left_shift(abnode *, int);
        void right_shift(abnode *, int);
        void fuse(abnode *, int);
//        abnode *search(int, abnode *, int *);
    public:
        ABTree(int a, int b) {
            // Number of children -> number of keys
            A = a-1;
            B = b-1;
            root = nullptr;
        }
        void insert(int, int *, int *);
        void del(int, int *, int *);
        void display(abnode *node = nullptr);
        void clean(abnode *node = nullptr);
};


#endif //ABTREE_ABTREE_H
