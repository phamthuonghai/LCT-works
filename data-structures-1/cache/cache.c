#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

int DEBUG = 0;
int TRIVIAL = 0;
int SIM = 0;
int *a;
int n;

clock_t swap(int r1, int c1, int r2, int c2) {
    if (SIM) {
        printf("X %d %d %d %d\n", r1, c1, r2, c2);
        return 0;
    }

    clock_t before = clock();
    int id1 = r1*n + c1, id2 = r2*n + c2;
    int tmp = *(a+id1);
    *(a+id1) = *(a + id2);
    *(a + id2) = tmp;
    return clock() - before;
}

clock_t trivial_transpose(int n) {
    long long int cnt = 0;
    for (int i=0; i<n; ++i) {
        for (int j=i+1;j<n; ++j) {
            cnt += swap(i, j, j, i);
        }
    }
    return cnt;
}

clock_t transpose_and_swap(int rA1, int cA1, int rA2, int cA2, int rB1, int cB1, int rB2, int cB2) {
    long long int cnt = 0;

    // Input indices are NOT always of square
    if (rA2-rA1 == 1 || cA2-cA1 == 1) {
        if (rA2-rA1 == 1 && cA2-cA1 == 1) {
            return swap(rA1, cA1, rB1, cB1);
        } else if (rA2-rA1 == 2) {
            cnt = swap(rA1, cA1, rB1, cB1);
            return cnt + swap(rA2-1, cA1, rB1, cB2-1);
        } else {
            cnt = swap(rA1, cA1, rB1, cB1);
            return cnt + swap(rA1, cA2-1, rB2-1, cB1);
        }
    }
    int rAm = (rA1+rA2)/2, cAm = (cA1+cA2)/2, rBm = (rB1+rB2)/2, cBm = (cB1+cB2)/2;
    cnt = transpose_and_swap(rA1, cA1, rAm, cAm, rB1, cB1, rBm, cBm);   // A11, B11
    cnt += transpose_and_swap(rA1, cAm, rAm, cA2, rBm, cB1, rB2, cBm);  // A12, B21
    cnt += transpose_and_swap(rAm, cA1, rA2, cAm, rB1, cBm, rBm, cB2);  // A21, B12
    cnt += transpose_and_swap(rAm, cAm, rA2, cA2, rBm, cBm, rB2, cB2);  // A22, B22
    return cnt;
}

clock_t transpose(int r1, int c1, int r2, int c2) {
    // Input indices are always of square
    if (r2-r1 == 1)
        return 0;
    else if (r2-r1 == 2) {
        return swap(r1, c2-1, r2-1, c1);
    }
    int rm = (r1+r2)/2, cm = (c1+c2)/2;
    long long cnt = 0;
    cnt += transpose(r1, c1, rm, cm);                          // A11
    cnt += transpose(rm, cm, r2, c2);                          // A22
    cnt += transpose_and_swap(r1, cm, rm, c2, rm, c1, r2, cm); // A12, A21
    return cnt;
}

void turn_params(char *param, int argc, char **argv, int *var) {
    for (int i=1; i<argc; i++)
        if (strcmp(argv[i], param) == 0) {
            *var = 1;
            return;
        }
}

/*
 * Command line parameters:
 * file_name [options]
 * Options:
 * -d: debug mode
 * -t: use trivial transpose
 * -s: use simulator
 *
 * Ex:
 * + Run with simulator:    cache cache.log -s | cachesim 64 64 > cachesim.log
 * + Run in real computer:  cache cache.log
 */
int main(int argc, char **argv) {
    turn_params("-d", argc, argv, &DEBUG);
    turn_params("-t", argc, argv, &TRIVIAL);
    turn_params("-s", argc, argv, &SIM);
    float k = 54;
    double msec;
    long long int swap_cnt;
    FILE *f;
    while (1) {
        n = (int)ceil(pow(2, k/9));
        if (SIM)
            printf("N %d\n", n);
        else
            a = (int *)malloc(n * n * sizeof(int));
        if (TRIVIAL)
            msec = trivial_transpose(n);
        else
            msec = transpose(0, 0, n, n);
        msec = (msec * 1000 / CLOCKS_PER_SEC);
        swap_cnt = n*(n-1)/2;
        f = fopen(argv[1], "a");
        fprintf(f, "%d,%lld,%f,%f\n", n, swap_cnt, msec, msec/swap_cnt);
        fclose(f);
        if (SIM)
            printf("E\n");
        else
            free(a);
        k++;
    }
    return 0;
}