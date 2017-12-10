#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Switch on if you wish to print detailed debugging messages
#if 0
#define DEBUG(...) printf(__VA_ARGS__)
#else
#define DEBUG(...) do { } while (0)
#endif
// By default, the matrix is stored in the memory row-by-row.
// If you define ZORDER, the recursive Z-order will be used instead.
#undef ZORDER
static int lino;
static int N;
static int *array;
static int block_size;
static unsigned int block_shift;
static int memory_blocks;
static int cache_blocks;
struct cache_node {
    struct cache_node *next, *prev;
    int block;
};
static struct cache_node lru_head;
static struct cache_node *cache_nodes;
static struct cache_node **block_to_cache;
static long long int cache_accesses, cache_misses;
static void error(char *msg)
{
    fprintf(stderr, "Error on line %d: %s\n", lino, msg);
    exit(1);
}
static void *xmalloc(int size)
{
    void *p = malloc(size);
    if (!p)
        error("Out of memory");
    return p;
}
static int pos(int i, int j)
{
#ifdef ZORDER
    int out = 0;
  for (int b=0; (1<<b) < N; b++)
    {
      if (i & (1 << b))
	out |= 1 << (2*b);
      if (j & (1 << b))
	out |= 2 << (2*b);
    }
  return out;
#else
    return i*N + j;
#endif
}
static void matrix_init(void)
{
    if (N < 4)
        error("Matrices smaller than 4x4 are not supported");
    array = xmalloc(N*N*sizeof(int));
    printf("%d", N);
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            array[pos(i,j)] = i*N + j + 42;
}
static void matrix_cleanup(void)
{
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            if (array[pos(i,j)] != j*N + i + 42)
                error("Wrong answer");
    free(array);
}
static void cache_init(void)
{
    if (block_size < 4 || (block_size & (block_size - 1)))
        error("Block size must be at least 4 and a power of 2");
    if (cache_blocks < 2)
        error("Cache must contain at least 2 blocks");
    block_shift = 1;
    while ((1U << block_shift) < (unsigned int) block_size)
        block_shift++;
    memory_blocks = (N*N*sizeof(int) + block_size - 1) / block_size;
    block_to_cache = xmalloc(memory_blocks * sizeof(struct cache_node *));
    for (int i=0; i<memory_blocks; i++)
        block_to_cache[i] = NULL;
    lru_head.next = lru_head.prev = &lru_head;
    cache_nodes = xmalloc(cache_blocks * sizeof(struct cache_node));
    for (int i=0; i<cache_blocks; i++)
    {
        struct cache_node *n = &cache_nodes[i];
        n->next = lru_head.next;
        n->prev = &lru_head;
        n->block = -1;
        n->prev->next = n;
        n->next->prev = n;
    }
    cache_accesses = 0;
    cache_misses = 0;
    DEBUG("Initialized cache: block_size=%d block_shift=%u memory_blocks=%d cache_blocks=%d\n",
          block_size, block_shift, memory_blocks, cache_blocks);
}
static void cache_access(int i)
{
    int b = (4*i) >> block_shift;
    if (b < 0 || b >= memory_blocks)
        error("Internal error: cache_access out of cache");
    cache_accesses++;
    struct cache_node *n = block_to_cache[b];
    if (!n)
    {
        // Not in cache: take the last entry from LRU and evict it
        n = lru_head.prev;
        if (n->block >= 0)
        {
            DEBUG("Cache #%d: reading %d, evicting %d\n", (int)(n - cache_nodes), b, n->block);
            block_to_cache[n->block] = NULL;
        }
        else
            DEBUG("Cache #%d: reading %d\n", (int)(n - cache_nodes), b);
        block_to_cache[b] = n;
        n->block = b;
        cache_misses++;
    }
    else
        DEBUG("Cache #%d: touching %d\n", (int)(n - cache_nodes), b);
    // Remove from LRU list
    struct cache_node *prev = n->prev;
    struct cache_node *next = n->next;
    prev->next = next;
    next->prev = prev;
    // Add at the head of LRU
    n->next = lru_head.next;
    n->prev = &lru_head;
    n->next->prev = n;
    n->prev->next = n;
}
static void cache_cleanup(void)
{
    free(block_to_cache);
    free(cache_nodes);
    printf(",%lld,%lld,%lld\n", cache_accesses, cache_misses, cache_misses * block_size);
    fflush(stdout);
}
static int rd(int i, int j)
{
    if (i < 0 || i >= N || j < 0 || j >= N)
        error("Position out of matrix");
    int p = pos(i,j);
    cache_access(p);
    return array[p];
}
static void wr(int i, int j, int x)
{
    if (i < 0 || i >= N || j < 0 || j >= N)
        error("Position out of matrix");
    array[pos(i,j)] = x;
}
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <block-size-in-B> <cache-size-in-blocks>\n", argv[0]);
        return 1;
    }
    block_size = atoi(argv[1]);
    cache_blocks = atoi(argv[2]);
    char line[256];
    while (fgets(line, sizeof(line), stdin))
    {
        lino++;
        char *nl = strchr(line, '\n');
        if (!nl)
            error("Line not terminated or too long");
        *nl = 0;
        switch (line[0])
        {
            case 'N':
                if (N)
                    error("Missing E command");
                if (line[1] != ' ' || sscanf(line+2, "%d", &N) != 1)
                    error("Matrix size expected");
                matrix_init();
                cache_init();
                break;
            case 'X':
            {
                int i1, j1, i2, j2;
                if (line[1] != ' ' || sscanf(line+2, "%d%d%d%d", &i1, &j1, &i2, &j2) != 4)
                    error("4 arguments expected");
                if (!N)
                    error("Missing N command");
                int t = rd(i1, j1);
                wr(i1, j1, rd(i2, j2));
                wr(i2, j2, t);
                break;
            }
            case 'E':
                if (!N)
                    error("Misplaced E command");
                matrix_cleanup();
                cache_cleanup();
                printf("\n");
                N = 0;
                break;
            default:
                error("Unrecognized operation");
        }
    }
    if (N)
        error("Missing E command");
    return 0;
}