#include <cstdio>
#include <cstdlib>
#include "abtree.h"

//#define DEBUG 1

/* command line: abtree [a] [b] */
int main(int argc, char *argv[]) {
#ifdef DEBUG
    FILE *fp = freopen("../bia.txt", "r", stdin);
#endif
    char line[50], command;
    int n = 0, st_visit, st_edit, k, cnt_ops = 0;
    float sum_visit = 0, sum_edit = 0;

    ABTree *tree = new ABTree(atoi(argv[1]), atoi(argv[2]));

    while (fgets(line, sizeof(line), stdin) != nullptr)
    {
        if (line[0] == '#')
        {
            if (n != 0)
            {
                tree->clean();
                printf("%d,%f,%f\n", n, sum_visit / cnt_ops, sum_edit / cnt_ops);
                sum_edit = 0;
                sum_visit = 0;
                cnt_ops = 0;
            }

            sscanf(line, "%c %d", &command, &n);
        }
        else if (line[0] == 'I')
        {
            sscanf(line, "%c %d", &command, &k);
            st_edit = 0;
            st_visit = 0;
            tree->insert(k, &st_edit, &st_visit);
            sum_edit += st_edit;
            sum_visit += st_visit;
            cnt_ops++;
#ifdef DEBUG
            printf("\n%s %d %d\n", line, st_edit, st_visit);
            tree->display();
#endif
        }
        else if (line[0] == 'D')
        {
            sscanf(line, "%c %d", &command, &k);
            st_edit = 0;
            st_visit = 0;
            tree->del(k, &st_edit, &st_visit);
            sum_edit += st_edit;
            sum_visit += st_visit;
            cnt_ops++;
#ifdef DEBUG
            printf("\n%s %d %d\n", line, st_edit, st_visit);
            tree->display();
#endif
        }
    }

#ifdef DEBUG
    tree->display() ;
    fclose(fp);
#endif
    return 0;
}