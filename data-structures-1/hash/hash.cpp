#include <iostream>
#include <time.h>
#include <inttypes.h>

enum SchemeType {
    LINEARPROB,
    CUCKOO
};

enum FunctionType {
    TAB,
    MULTI_SHIFT,
    NAIVE
};


/*
 * This is the xoroshiro128+ random generator, designed in 2016 by David Blackman
 * and Sebastiano Vigna, distributed under the CC-0 license. For more details,
 * see http://vigna.di.unimi.it/xorshift/.
 */
static uint64_t rng_state[2];

static uint64_t rng_rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next_u64()
{
    uint64_t s0 = rng_state[0], s1 = rng_state[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_state[0] = rng_rotl(s0, 55) ^ s1 ^ (s1 << 14);
    rng_state[1] = rng_rotl(s1, 36);
    return result;
}

static void rng_setup(unsigned int seed)
{
    rng_state[0] = seed * 0xdeadbeef;
    rng_state[1] = seed ^ 0xc0de1234;
    for (int i=0; i<100; i++)
        rng_next_u64();
}

class HashTable {
    private:
        SchemeType scheme;
        FunctionType func;
        int log_m;
        uint64_t* table;
        uint64_t at[2][8];
        uint64_t a[2];
        uint64_t t_hash_func(uint64_t x, int, int);
        uint64_t hash_func(uint64_t x, int id=0);
        uint64_t rehash();
    public:
        uint64_t m;
        uint64_t n;
        uint64_t rehash_cnt;
        // All experiments deal with m=2^k, restricts the implementation to this constraint to speed up mod operations
        HashTable(SchemeType scheme, FunctionType func, int log_m = 20) {
            this->scheme = scheme;
            this->func = func;
            this->log_m = log_m;
            this->m = uint64_t(1 << log_m);

            this->table = new uint64_t[this->m];

            this->init();
        }
        void init();
        ~HashTable() {
            delete[] this->table;
        }
        uint64_t insert(uint64_t x);
        bool find(uint64_t x);
        double load_factor() {
            return this->n*1.0/this->m;
        }
};


void HashTable::init() {
    this->n = 0;
    this->a[0] = rng_next_u64();
    this->a[1] = rng_next_u64();
    for (int i=0; i<8; ++i) {
        this->at[0][i] = rng_next_u64();
        this->at[1][i] = rng_next_u64();
    }
    for (int i=0; i<m; ++i)
        this->table[i] = 0; // use 0 for empty slot, hence can't accept 0 as an input
}


bool HashTable::find(uint64_t x) {
    uint64_t i = this->hash_func(x);
    if (this->scheme == LINEARPROB) {
        while (this->table[i] != 0) {
            if (this->table[i] == x)
                return true;
            i = (i + 1) & (m - 1);
        }
    }
    else if (this->scheme == CUCKOO) {
        if (this->table[i] == x)
            return true;
        i = this->hash_func(x, 1);
        if (this->table[i] == x)
            return true;
    }
    else {
        std::cout << "Hashing scheme not found!";
        exit(1);
    }
    return false;
}

uint64_t HashTable::insert(uint64_t x) {
    if (this->find(x))
        return 0;

    uint64_t i = this->hash_func(x);
    uint64_t res = 0;

    if (this->scheme == LINEARPROB) {
        while (this->table[i] != 0) {
            i = (i+1) & (m-1);
            res++;
        }
    }
    else if (this->scheme == CUCKOO) {
        uint64_t tmp;
        bool is_inserted = false;
        while (not is_inserted) {
            for (uint64_t nTrial=1; nTrial <= this->m; ++nTrial) {
                if (this->table[i] == 0) {
                    is_inserted = true;
                    break;
                }
                tmp = x; x = this->table[i]; this->table[i] = tmp;
                res++;
                i = this->hash_func(x, static_cast<int>(nTrial & 1));
            }
            if (not is_inserted) {
                res += this->rehash();
                if (this->rehash_cnt > this->log_m) {
                    return res;
                }
            }
        }
    }
    else {
        std::cout << "Hashing scheme not found!";
        exit(1);
    }

    this->n++;
    this->table[i] = x;

    return res;
}


uint64_t HashTable::t_hash_func(uint64_t x, int id1, int id2) {
    // Implement Ti as multiply-mod function
    return (this->at[id1][id2] * ((x >> (id2*8)) & 0xff)) & (this->m-1);
}


uint64_t HashTable::hash_func(uint64_t x, int id) {
    if (this->func == NAIVE) {
        return x & (this->m-1);
    }
    else if (this->func == MULTI_SHIFT) {
        return (this->a[id]*x) >> (64-this->log_m);
    }
    else if (this->func == TAB){
        // Split x into 8 parts, each has 8 bits
        return this->t_hash_func(x, id, 0) \
            ^ this->t_hash_func(x, id, 1) \
            ^ this->t_hash_func(x, id, 2) \
            ^ this->t_hash_func(x, id, 3) \
            ^ this->t_hash_func(x, id, 4) \
            ^ this->t_hash_func(x, id, 5) \
            ^ this->t_hash_func(x, id, 6) \
            ^ this->t_hash_func(x, id, 7);
    }
    else {
        std::cout << "Hash function type not found!";
        exit(1);
    }
}

uint64_t HashTable::rehash() {
    this->rehash_cnt++;
    uint64_t *tmp = new uint64_t[this->m];
    uint64_t cnt = 0;
    uint64_t res = 0;

    for (int i=0; i<m; ++i)
        if (this->table[i] != 0)
            tmp[cnt++] = this->table[i];

    this->init();

    for (int i=0; i<cnt; ++i) {
        res += this->insert(tmp[i]);
    }

    delete[] tmp;
    return res;
}

/*
 * Command line parameters:
 * hash [task_number] [scheme] [func]
 *      task_number: 0 (1st task), 1 (2nd task)
 *      scheme: 0 (Linear probing), 1 (cuckoo)
 *      func: 0 (tabulation), 1 (multiply-shift), 2 (naive)
 *
 * Ex:
 * + hash 0 1 0 // 1st task, cuckoo hashing with tabulation
 */
int main(int argc, char** argv) {
    rng_setup(69);
    int scheme, func, task, log_m = 20;
    sscanf(argv[1], "%d", &task);
    sscanf(argv[2], "%d", &scheme);
    sscanf(argv[3], "%d", &func);

    if (scheme == CUCKOO && func == NAIVE) {
        std::cout << "Scheme and function are not compatible!";
        exit(1);
    }

    if (task == 1 && (scheme == CUCKOO || func == NAIVE)) {
        std::cout << "Wrong arguments for 2nd task!";
        exit(1);
    }

    uint64_t t;
    uint64_t res;
    HashTable *ht;

    clock_t before, diff;
    long double nsec;

    int cnt = 0;

    if (task == 0) {
        ht = new HashTable((SchemeType)scheme, (FunctionType)func);

        while (true) {
            t = rng_next_u64();
            if (t == 0)
                continue;
            ht->rehash_cnt = 0;

            before = clock();
            res = ht->insert(t);
            diff = clock() - before;
            nsec = (diff * 1000000.0 / CLOCKS_PER_SEC);

            printf("%.2f,%" PRIu64 ",%Lf\n", ht->load_factor(), res, nsec);

            // Criteria to stop experiment
            if ((scheme == CUCKOO && ht->rehash_cnt > log_m) or (scheme == LINEARPROB && ht->load_factor() > 0.99)) {
                ht->init();
                cnt++;
                // Repeat experiment 50 times
                if (cnt >= 50)
                    break;
            }
        }

        delete ht;
    }
    else if (task == 1) {
        for (log_m=20; log_m<=64; log_m++){
            ht = new HashTable((SchemeType)scheme, (FunctionType)func, log_m);

            uint64_t entry, p089 = (uint64_t)(0.89*ht->m), p091 = (uint64_t)(0.91*ht->m);

            // Repeat experiment 50 times
            for (cnt=0; cnt<50; ++cnt) {
                ht->init();
                res = 0;
                float res_cnt = 0;
                for (entry=1; entry<p089; ++entry)
                    ht->insert(entry);

                for (entry=p089; entry<=p091; ++entry) {
                    res += ht->insert(entry);
                    res_cnt += 1;
                }

                printf("%d,%" PRIu64 ",%d,%f\n", log_m, ht->m, (int)res_cnt, res/res_cnt);
                fflush(stdout);
            }

            delete ht;
        }
    }
    else {
        std::cout << "Task " << argv[1] <<  " not found!";
    }

    return 0;
}