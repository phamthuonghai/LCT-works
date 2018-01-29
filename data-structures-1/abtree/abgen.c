#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;

static void usage(void)
{
  fprintf(stderr, "Usage: gen [-s <student-id>] [-n <elements>] [-b]\n");
  exit(1);
}

int student_id = -1;
int num_elements = -1;
int biased;

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

static uint64_t rng_next_u64(void)
{
	uint64_t s0 = rng_state[0], s1 = rng_state[1];
	uint64_t result = s0 + s1;
	s1 ^= s0;
	rng_state[0] = rng_rotl(s0, 55) ^ s1 ^ (s1 << 14);
	rng_state[1] = rng_rotl(s1, 36);
	return result;
}

static uint32_t rng_next_u32(void)
{
	return rng_next_u64() >> 11;
}

static void rng_setup(unsigned int seed)
{
	rng_state[0] = seed * 0xdeadbeef;
	rng_state[1] = seed ^ 0xc0de1234;
	for (int i=0; i<100; i++)
		rng_next_u64();
}

/*** The RNG ends here ***/

void permute(u64 *p, int n)
{
  for (int i=0; i<n-1; i++)
    {
      int j = i + rng_next_u32() % (n-i);
      u64 x = p[i];
      p[i] = p[j];
      p[j] = x;
    }
}

void generate_std(int n)
{
  printf("# %d\n", n);

  u64 *p[2] = { malloc(n * sizeof(u64)), malloc(n * sizeof(u64)) };
  int passes = 10;

  for (int pass=0; pass<passes; pass++)
    {
      u64 *new = p[pass % 2];
      u64 *old = p[1 - pass % 2];

      for (int i=0; i<n; i++)
	new[i] = (passes+1)*i + pass + 1;
      permute(new, n);
      permute(old, n);

      for (int i=0; i<n; i++)
	{
	  if (pass)
	    printf("D %ju\n", (uintmax_t) old[i]);
	  if (pass < passes-1)
	    printf("I %ju\n", (uintmax_t) new[i]);
	}
    }

  free(p[0]);
  free(p[1]);
}

void generate_biased(int n)
{
  int q = 1;
  int s = 0;
  while (s+q < n/10)
    {
      s += q;
      q *= 3;
    }
  s = 2*s;
  n = s * ((n+s-1) / s);
  printf("# %d\n", n);

  u64 *p = malloc(n * sizeof(u64));
  for (int i=0; i<n; i++)
    p[i] = i + 1;
  for (int i=0; i<n; i++)
    printf("I %ju\n", (uintmax_t) p[i]);

  int want = 10*n;
  int ops = 0;
  while (ops < want)
    {
      for (int pass=0; pass<2; pass++)
	{
	  int x = s/2+1;
	  while (x < n)
	    {
	      if (!pass)
		printf("D %ju\n", (uintmax_t) p[x]);
	      else
		printf("I %ju\n", (uintmax_t) p[x]);
	      x += s + 1;
	      ops++;
	    }
	}
    }

  for (int i=0; i<n; i++)
    printf("D %ju\n", (uintmax_t) p[i]);

  free(p);
}

void generate(int n)
{
  if (biased)
    generate_biased(n);
  else
    generate_std(n);
}

int main(int argc, char **argv)
{
  int opt;

  if (argc > 1 && !strcmp(argv[1], "--help"))
    usage();

  while ((opt = getopt(argc, argv, "bn:s:")) >= 0)
    switch (opt)
      {
      case 's':
	student_id = atoi(optarg);
	break;
      case 'n':
	num_elements = atoi(optarg);
	break;
      case 'b':
	biased = 1;
	break;
      default:
	usage();
      }

  if (student_id < 0)
    {
      fprintf(stderr, "WARNING: Student ID not given, defaulting to 42.\n");
      student_id = 42;
    }
  rng_setup(student_id);

  if (num_elements >= 0)
    {
      if (num_elements < 20)
	{
	  fprintf(stderr, "ERROR: You must ask for at least 20 elements\n");
	  exit(1);
	}
      generate(num_elements);
    }
  else
    {
      for (int e=20; e <= 40; e++)
	generate(pow(2, e/2. + .01));
    }

  return 0;
}
