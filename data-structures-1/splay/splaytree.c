//
// Created by phamthuonghai on 20/10/17.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int KEY_TYPE;

typedef struct node{
    KEY_TYPE key;
    struct node *left;
    struct node *right;
} node;

int DEBUG = 0;
int NAIVE = 0;

/*
        u                   x
       / \                 / \
      x   C     ==>       A   u
     / \                     / \
    A   B                   B   C
*/
node *rotate_right(node *u)
{
    node *x = u->left;
    u->left = x->right;
    x->right = u;
    return x;
}

/*
       u                    x
      / \                  / \
     A   x      ==>       u   C
        / \              / \
       B   C            A   B
 */
node *rotate_left(node *u)
{
    node *x = u->right;
    u->right = x->left;
    x->left = u;
    return x;
}

node *new_node(KEY_TYPE k)
{
    node *x = (node *)malloc(sizeof(node));
    x->key   = k;
    x->left  = x->right  = NULL;
    return x;
}

// Splay the found node to granddad position,
// if node with key not found, then splay the node with closest key
int splay_classical(node **p_g, KEY_TYPE k)
{
    node *g = *p_g;
    if (g == NULL)
        return 0;
    if (g->key == k)
        return 1;

    int tmp_dep = 0;

    if (g->key > k)
    {
        if (g->left == NULL) return 0;

        if (g->left->key > k)                              // zig-zig
        {
            tmp_dep = splay_classical(&(g->left->left), k);
            g = rotate_right(g);
        }
        else if (g->left->key < k)                         // zig-zag
        {
            tmp_dep = splay_classical(&(g->left->right), k);

            if (g->left->right != NULL)
                g->left = rotate_left(g->left);
        }

        // This handles the second rotation or the single zig
        if (g->left != NULL)
            g = rotate_right(g);
    }
    else
    {
        if (g->right == NULL) return 0;

        if (g->right->key > k)                             // zag-zig
        {
            tmp_dep = splay_classical(&(g->right->left), k);

            if (g->right->left != NULL)
                g->right = rotate_right(g->right);
        }
        else if (g->right->key < k)                        // zag-zag
        {
            tmp_dep = splay_classical(&(g->right->right), k);
            g = rotate_left(g);
        }

        // This handles the second rotation or the single zag
        if (g->right != NULL)
            g = rotate_left(g);
    }

    *p_g = g;
    return tmp_dep + 2;
}

int splay_naive(node **p_g, KEY_TYPE k)
{
    node *g = *p_g;
    int tmp_dep = 0;

    if (g == NULL)
        return 0;
    if (g->key == k)
        return 1;

    if (g->key > k)
    {
        tmp_dep = splay_naive(&(g->left), k);
        if (g->left != NULL)
            g = rotate_right(g);
    }
    else
    {
        tmp_dep = splay_naive(&(g->right), k);
        if (g->right != NULL)
            g = rotate_left(g);
    }

    *p_g = g;
    return tmp_dep + 1;
}

int splay(node **p_g, KEY_TYPE k)
{
    return (NAIVE == 0)? splay_classical(p_g, k): splay_naive(p_g, k);
}

int search(node **root, KEY_TYPE k)
{
    // Simply splay the node with the key to the root
    return splay(root, k);
}

void insert(node **p_root, KEY_TYPE k)
{

    if (*p_root == NULL)
    {
        *p_root = new_node(k);
        return;
    }

    // Find u closest to x (key: k)
    splay(p_root, k);
    node *u = *p_root;

    if (u->key == k) return;

    // Insert x to the tree
    node *x  = new_node(k);
    if (u->key > k)
    {
        x->right = u;
        x->left = u->left;
        u->left = NULL;
    }
    else
    {
        x->left = u;
        x->right = u->right;
        u->right = NULL;
    }

    *p_root = x;
}

void turn_params(char *param, int argc, char **argv, int *var)
{
    for (int i=1; i<argc; i++)
        if (strcmp(argv[i], param) == 0)
        {
            *var = 1;
            return;
        }
}

/*
 * Command line parameters:
 * -d: debug mode
 * -n: use naive tree
 *
 * Ex:
 * + Run classical Splay tree:  splaygen -s 69 -t 10 | splaytree
 * + Run naive Splay tree:      splaygen -s 69 -t 10 | splaytree -n
 */
int main(int argc, char **argv)
{
    turn_params("-d", argc, argv, &DEBUG);
    turn_params("-n", argc, argv, &NAIVE);

    char line[50];
    int n = 0, dep, search_count = 0;
    float dep_total = 0;
    char command;
    KEY_TYPE k;
    node *root = NULL;

    while (fgets(line, sizeof(line), stdin) != NULL)
    {
        if (line[0] == '#')
        {
            if (n != 0)
            {
                if (search_count == 0)
                    printf("%d,0\n", n);
                else
                    printf("%d,%f\n", n, dep_total/search_count);
            }

            root = NULL;
            dep_total = 0;
            search_count = 0;
            sscanf(line, "%c %d", &command, &n);
        }
        else if (line[0] == 'I')
        {
            sscanf(line, "%c %d", &command, &k);
            insert(&root, k);
        }
        else if (line[0] == 'F')
        {
            sscanf(line, "%c %d", &command, &k);
            dep = search(&root, k);
            if (DEBUG)
                printf("%d\n", dep);
            dep_total += dep;
            search_count += 1;
        }
    }


    if (search_count == 0)
        printf("%d,0\n", n);
    else
        printf("%d,%f\n", n, dep_total/search_count);

    return 0;
}
