#include <cstdlib>
#include <cstdio>
#include "abtree.h"

/* insert a key in the tree*/
void ABTree::insert(int key, int *st_edit, int *st_visit) {
    int i;
    abnode *c, *node;

    if (insert_node(key, root, &i, &c, st_edit, st_visit)) {    // grow upwards
        node = (abnode *)malloc(sizeof(abnode));
        node->count = 1;
        node->keys[1] = i;
        node->child[0] = root;
        node->child[1] = c;
        root = node;
        (*st_edit)++;
    }
}

/* insert the key in the tree started with node
 * return true if tree needed to grow upwards &(node, new key k1, new child c)
 * */
bool ABTree::insert_node(int key, abnode *node, int *k1, abnode **c, int *st_edit, int *st_visit) {
    int pos;
    if (node == nullptr) {
        *k1 = key;
        *c = nullptr;
        return true;
    }
    else {
        (*st_visit)++;
        if (!search(key, node, &pos)) {                      // ignore existed key
            if (insert_node(key, node->child[pos], k1, c, st_edit, st_visit)) {
                // if key k1 is pushed up from children
                if (node->count < B) {
                    set_key(*k1, *c, node, pos);                  // safe to set k1 & c in
                    (*st_edit)++;
                    return false;
                } else {
                    split(node, pos, k1, c);
                    (*st_edit) += 2;                              // split into 2 nodes
                    return true;
                }
            }
        }
        return false;
    }
}

// search key in the tree with node as root
// * return node contains key &(pos)
// * */
//abnode *ABTree::search(int key, abnode *node, int *pos) {
//    if (node == nullptr)
//        return nullptr;
//    else {
//        if (search_node(key, node, pos))
//            return node;
//        else
//            return search(key, node->child[*pos], pos);
//    }
//}

/* search for key inside the node
 * return true if found (&pos)
 * */
bool ABTree::search(int key, abnode *node, int *pos) {
    if (key < node->keys[1]) {
        *pos = 0;
        return false;
    }
    *pos = node->count;
    while ((key < node->keys[*pos]) && (*pos > 1))
        (*pos)--;
    return key == node->keys[*pos];
}


/* set the key and child c to position pos of the node */
void ABTree::set_key(int key, abnode *c, abnode *node, int pos) {
    // Move all elements after pos 1 step right
    for (int i=node->count; i>pos; i--) {
        node->keys[i+1] = node->keys[i];
        node->child[i+1] = node->child[i];
    }

    node->keys[pos+1] = key;
    node->child[pos+1] = c;
    node->count++;
}

/* remove the key from node */
void ABTree::clear_key(abnode *node, int pos) {
    // Move all elements after pos 1 step left
    for (int i=pos+1; i<=node->count; i++) {
        node->keys[i-1] = node->keys[i];
        node->child[i-1] = node->child[i];
    }
    node->count--;
}

/* split the node */
void ABTree::split(abnode *node, int pos, int *new_key, abnode **new_node) {
    // Input key & child node, then used as output afterwards, just to reduce number of params
    int key = *new_key;
    abnode *c = *new_node;

    int mid = (pos<=A)?A:A+1;

    *new_node = (abnode *) malloc(sizeof(abnode));

    // Move half of the children to new node
    for (int i=mid+1; i<=B; i++)
    {
        (*new_node)->keys[i-mid] = node->keys[i];
        (*new_node)->child[i-mid] = node->child[i];
    }

    (*new_node)->count = B-mid;
    node->count = mid;

    if (pos <= A)
        set_key(key, c, node, pos);
    else
        set_key(key, c, *new_node, pos-mid);

    *new_key = node->keys[node->count];
    (*new_node)->child[0] = node->child[node->count];
    node->count--;
}

/* delete key from the tree */
void ABTree::del(int key, int *st_edit, int *st_visit)
{
    abnode * tmp;
    if (del_node(key, root, st_edit, st_visit)) {
        if (root->count == 0) {
            // Shrink tree down
            tmp = root;
            root = root->child[0];
            free(tmp);
            (*st_edit)++;
        }
    }
}

/* delete key from a tree started with node */
bool ABTree::del_node(int key, abnode *node, int *st_edit, int *st_visit)
{
    int pos;
    bool flag;
    if (node == nullptr)
        return false;
    else {
        flag = search(key, node, &pos);
        (*st_visit)++;
        if (flag) {
            if (node->child[pos-1]) {
                copy_succ(node, pos);                                   // replace deleted node by successor
                // count visits on the way down to leaf node instead of copy_succ
                flag = del_node(node->keys[pos], node->child[pos], st_edit, st_visit);
            }
            else
                clear_key(node, pos);                                   // leaf node, safe to delete
            (*st_edit)++;
        }
        else
            flag = del_node(key, node->child[pos], st_edit, st_visit);  // key is not in this node

        if (node->child[pos] != nullptr) {
            if (node->child[pos]->count < A) {
                restructure(node, pos);
                (*st_edit) += 2;
                (*st_visit) += 1; // one has been counted before
            }
        }
        return flag;
    }
}

/* copy the successor of the key in pos of node */
void ABTree::copy_succ(abnode *node, int pos) {
    abnode *tmp = node->child[pos];

    while (tmp->child[0])
        tmp = tmp->child[0];

    node->keys[pos] = tmp->keys[1];
}

/* restructure the tree started with node */
void ABTree::restructure(abnode *node, int pos) {
    if (pos == 0) {                                     // consider only right sibling
        if (node->child[1]->count > A)
            left_shift(node, 1);
        else
            fuse(node, 1);
    } else if (pos == node->count) {                    // consider only left sibling
        if (node->child[pos-1]->count > A)
            right_shift(node, pos);
        else
            fuse(node, pos);
    } else {
        if (node->child[pos-1]->count > A)
            right_shift(node, pos);
        else if (node->child[pos+1]->count > A)
            left_shift(node, pos+1);
        else
            fuse(node, pos);
    }
}

/* shift one key from node->child[pos] to node->child[pos-1] */
void ABTree::left_shift(abnode *node, int pos) {
    abnode *tmp;

    tmp = node->child[pos-1];
    tmp->count++;
    tmp->keys[tmp->count] = node->keys[pos];                // move key from node to node->child[pos-1]
    tmp->child[tmp->count] = node->child[pos]->child[0];

    tmp = node->child[pos];
    node->keys[pos] = tmp->keys[1];                         // move key from node->child[pos] to node
    tmp->child[0] = tmp->child[1];
    tmp->count--;

    for (int i=1; i<=tmp->count; i++) {
        tmp->keys[i] = tmp->keys[i+1];
        tmp->child[i] = tmp->child[i+1];
    }
}

/* shift one key from node->child[pos-1] to node->child[pos] */
void ABTree::right_shift(abnode *node, int pos) {
    abnode *tmp = node->child[pos];

    for (int i=tmp->count; i>0; i--) {
        tmp->keys[i+1] = tmp->keys[i];
        tmp->child[i+1] = tmp->child[i];
    }

    tmp->child[1] = tmp->child[0];
    tmp->count++;
    tmp->keys[1] = node->keys[pos];                        // move key from node to node->child[pos]

    tmp = node->child[pos-1];
    node->keys[pos] = tmp->keys[tmp->count];              // move key from node->child[pos-1] to node
    node->child[pos]->child[0] = tmp->child[tmp->count];
    tmp->count--;
}

/* merge node->child[pos] into node->child[pos-1] */
void ABTree::fuse(abnode *node, int pos) {
    abnode *tmp1, *tmp2;
    tmp1 = node->child[pos-1];
    tmp2 = node->child[pos];

    // move corresponding key from node to node->child[pos-1]
    tmp1->count++;
    tmp1->keys[tmp1->count] = node->keys[pos];

    // merge node->child[pos] into node->child[pos-1]
    tmp1->child[tmp1->count] = tmp2->child[0];

    for (int i=1; i<=tmp2->count; i++) {
        tmp1->count++;
        tmp1->keys[tmp1->count] = tmp2->keys[i];
        tmp1->child[tmp1->count] = tmp2->child[i];
    }

    for (int i=pos; i<node->count; i++) {
        node->keys[i] = node->keys[i+1];
        node->child[i] = node->child[i+1];
    }

    node->count--;
    free(tmp2);
}

/* print the tree to stdout */
void ABTree::display(abnode *node) {
    if (node == nullptr) {
        node = root;

        if (node == nullptr)
            return;

        printf("\n");
    }
    printf("( ");
    int i;
    for (i = 0; i<node->count; i++) {
        if (node->child[i] != nullptr)
            display(node->child[i]);
        printf("%d ", node->keys[i+1]);
    }
    if (node->child[i] != nullptr)
        display(node->child[i]);
    printf(") ");
}

/* free the whole tree started with node */
void ABTree::clean(abnode *node) {
    if (node == nullptr) {
        node = root;

        if (node == nullptr)
            return;
    }

    int i;
    for (i = 0; i<=node->count; i++) {
        if (node->child[i] != nullptr) {
            free(node->child[i]);
            node->child[i] = nullptr;
        }
    }

    free(node);
}
