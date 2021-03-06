//
// Created by phamthuonghai on 13/11/17.
//
#include <iostream>
#include <cmath>
#include <cstring>

bool DEBUG = false;
bool NAIVE = false;


typedef long int KEY_TYPE;

struct node
{
    KEY_TYPE key;       // priority
    long int eid;       // element id
    long int order;
    node *parent;
    node *child;
    node *left;
    node *right;
    bool marked;
};

class FibHeap
{
    private:
        long int nNodes;
    public:
        node *minH;
        FibHeap()
        {
            minH = nullptr;
            nNodes = 0;
        }
        ~FibHeap()
        {
            Init();
        }
        node *CreateNode(long int, KEY_TYPE);
        void InsertToList(node *, node *);
        void Insert(node *, bool);
        node *Link(node *, node *);
        long int Consolidate();
        long int ExtractMin();
        void Cut(node *, node *);
        void DecreaseKey(node *, KEY_TYPE);
        void PrintHeads();
        void Free(node *x = nullptr);
        void Init();
};

node* FibHeap::CreateNode(long int e, KEY_TYPE k)
{
    node *x;
    x = new node;
    x->eid = e;
    x->key = k;
    x->order = 0;
    x->parent = nullptr;
    x->child = nullptr;
    x->left = x;
    x->right = x;
    x->marked = false;
    return x;
}

void FibHeap::InsertToList(node *x, node *y)
{
    /*
     * Insert y to the right of x,
     * i.e. in the main list, to the begining of the list
     * which will lead to slightly different results in special test
     *
     * (x->right)->left = y;
     * y->left = x;
     * y->right = x->right;
     * x->right = y;
        */
    (x->left)->right= y;
    y->right = x;
    y->left = x->left;
    x->left = y;
}

void FibHeap::Insert(node* x, bool isNew = false)
{
    if (minH != nullptr)
    {
        // Insert x to the left of minH, order is not compulsory in tree list
        InsertToList(minH, x);

        if (x->key < minH->key)
            minH = x;
    }
    else
        minH = x;

    if (isNew)
        nNodes = nNodes + 1;
}

node *FibHeap::Link(node *x, node *y)
{
    node *tmp;
    if (x->key > y->key)
    {
        tmp = x;
        x = y;
        y = tmp;
    }

    // Remove y from its sibling list
    (y->left)->right = y->right;
    (y->right)->left = y->left;
    if (y->parent != nullptr)
    {
        if (y->parent->child == y)
        {
            if (y == y->left)
                y->parent->child = nullptr;
            else
                y->parent->child = y->left;
        }
    }

    if (x->child != nullptr)
    {
        InsertToList(x->child, y);
    }
    else
    {
        x->child = y;
        y->left = y;
        y->right = y;
    }

    y->parent = x;
    x->order++;
}

long int FibHeap::Consolidate()
{
    if (DEBUG)
    {
        printf("Consolidation start: ");
        PrintHeads();
    }

    if (minH == nullptr)
        return 0;

    long int d, i, D, nSteps = 0;
    bool isJoined = false;

    if (NAIVE)
        D = nNodes;
    else
        D = 2*((long int)log2(nNodes));

    node **A = new node*[D+1];
    for (i = 0; i <= D; i++)
        A[i] = nullptr;

    node *x = minH;
    while (true)
    {
        d = x->order;
        while (A[d] != nullptr and A[d] != x)
        {
            x = Link(x, A[d]);
            nSteps++;
            isJoined = true;

            minH = x;

            A[d] = nullptr;
            d = x->order;
        }

        A[d] = x;

        x = x->right;

        if (x == minH)
        {
            if (isJoined)
                isJoined = false;
            else
                break;
        }
    }

    delete[] A;

    x = minH;
    node *newMinH = minH;
    do
    {
        if (x->key < newMinH->key)
            newMinH = x;
        x = x->right;
    } while (x != minH);

    minH = newMinH;

    if (DEBUG)
    {
        printf("Consolidation end: ");
        PrintHeads();
    }

    return nSteps;
}

// Actually delete min node, return the number of steps
long int FibHeap::ExtractMin()
{
    node *memMinH = minH;
    long int nSteps = 0;

    if (minH->child == nullptr)
    {
        if (minH->left != minH) // Tree list contains > 1 tree
        {
            (minH->left)->right = minH->right;
            (minH->right)->left = minH->left;
            minH = minH->left;
        }
        else
        {
            minH = nullptr;
        }
    }
    else
    {
        node *child = minH->child;
        node *child2 = child->left;

        do {
            child->parent = nullptr;
            nSteps++;
            child = child->right;
        } while (child != minH->child);

        // Add children of minH to the tree list
        if (minH->left != minH) // Tree list contains > 1 tree
        {
            child2->right = minH->right;
            (minH->right)->left = child2;
            child->left = minH->left;
            (minH->left)->right= child;
        }

        minH = minH->child;
    }

    nNodes--;
    nSteps += Consolidate();
    delete memMinH;

    return nSteps;
}

void FibHeap::Cut(node* p, node* x)
{
    if (x == x->right)
    {
        p->child = nullptr;
    }
    else
    {
        (x->left)->right = x->right;
        (x->right)->left = x->left;
        if (p->child == x)
            p->child = x->left;
    }
    p->order--;

    x->parent = nullptr;
    x->marked = false;
    Insert(x);
}

void FibHeap::DecreaseKey(node *x, KEY_TYPE k)
{
    if (x == nullptr or x->key <= k)
        return;

    x->key = k;

    node *p = x->parent;
    while (p != nullptr)
    {
        Cut(p, x);
        if (not p->marked or NAIVE)
            break;
        x = p;
        p = x->parent;
    }

    if (p != nullptr and p->parent != nullptr)
        p->marked = true;
}

void FibHeap::PrintHeads()
{
    node* p = minH;
    if (p == nullptr)
    {
        printf("Empty!\n");
        return;
    }
    printf("Heap heads: ");
    do
    {
        printf("(%ld, %ld, %ld) ", p->eid, p->key, p->order);
        p = p->right;
    } while (p != minH);
    printf("\n");
}

void FibHeap::Free(node *x)
{
    if (x == nullptr)
        return;

    node *i = x;
    do
    {
        Free(i->child);
        delete i;
        i = i->right;
    } while (i != x);
}

void FibHeap::Init()
{
    if (minH != nullptr)
        Free(minH);

    minH = nullptr;
    nNodes = 0;
}

void turn_params(const char *param, int argc, char **argv, bool *var)
{
    for (int i=1; i<argc; i++)
        if (strcmp(argv[i], param) == 0)
        {
            *var = true;
            return;
        }
}

/*
 * Command line parameters:
 * -d: debug mode
 * -n: use naive tree
 *
 * Ex: to run on random test
 * + Run FibHeap:       heapgen -s 69 -r | fibheap
 * + Run naive FibHeap: heapgen -s 69 -r | fibheap -n
 */
int main(int argc, char **argv)
{
    turn_params("-d", argc, argv, &DEBUG);
    turn_params("-n", argc, argv, &NAIVE);

    if (DEBUG)
        freopen("data.txt", "r", stdin);

    char line[50];
    long int n = 0, e, nExtracts = 0, nOps = 0, tmp;
    double nSteps = 0;
    char command[3];
    KEY_TYPE k;

    FibHeap fh;
    node **el_ptrs = new node*[2000000];

    while (fgets(line, sizeof(line), stdin) != nullptr)
    {
        if (line[0] == '#') {
            if (n != 0)
            {
                if (nExtracts == 0)
                    printf("%ld,0,0,0\n", n);
                else
                    printf("%ld,%f,%f,%ld\n", n, nSteps/nExtracts, nSteps, nExtracts);
            }

            nSteps = 0;
            nExtracts = 0;
            nOps = 0;
            sscanf(line, "%s %ld", command, &n);
            fh.Init();
            for (long int i=0; i<n; ++i)
                el_ptrs[i] = nullptr;
        }
        else
        {
            switch (line[2])
            {
                case 'S':                                               // INS
                    sscanf(line, "%s %ld %ld", command, &e, &k);
                    el_ptrs[e] = fh.CreateNode(e, k);
                    fh.Insert(el_ptrs[e], true);
                    break;
                case 'C':                                               // DEC
                    sscanf(line, "%s %ld %ld", command, &e, &k);
                    fh.DecreaseKey(el_ptrs[e], k);
                    break;
                case 'L':                                               // DEL
                    if (fh.minH != nullptr)
                        el_ptrs[fh.minH->eid] = nullptr;
                    tmp = fh.ExtractMin();
                    nSteps += tmp;
                    if (DEBUG)
                        printf("%ld\n", tmp);
                    nExtracts++;
                    break;
                default:
                    break;
            }
            if (DEBUG)
            {
                nOps++;
                printf("%ld-%c-", nOps, line[2]);
                fh.PrintHeads();
            }
        }

    }

    if (nExtracts == 0)
        printf("%ld,0,0,0\n", n);
    else
        printf("%ld,%f,%f,%ld\n", n, nSteps/nExtracts, nSteps, nExtracts);


    delete[] el_ptrs;
    return 0;
}
