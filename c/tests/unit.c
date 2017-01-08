#include <stdio.h>
#include <math.h>
#include "../util.h"
#include "minunit.h"
#include "unit_compare.h"
#define W 28
#define H 28

int tests_run = 0;
char output_buf[1000];
int errors = 0;


static char* test_bslice_2d_1()
{
  uint8_t slice_2d_in[3] = {223,175,248};
  uint8_t slice_2d_out[2] = {239,128};
  uint8_t slice_2d_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_2d_1\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  int i;
  int x = 0;
  int y = 4;
  int w = 7;
  int h = 3;
  int kw = 3;
  int kh = 3;

  /* tests normal slice */
  bslice_2d(slice_2d_comp, slice_2d_in, x, y, w, h, kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, slice_2d_out[i], slice_2d_comp[i], i);
    mu_assert(output_buf, slice_2d_out[i] == slice_2d_comp[i]);
  }

  return 0;
}

static char* test_bslice_2d_2()
{
  uint8_t A_in[3] = {4,64,240};
  uint8_t C_actual[1] = {32};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_2d_2\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  int i;
  int x = -2;
  int y = 3;
  int w = 7;
  int h = 3;
  int kw = 3;
  int kh = 3;

  /* tests padding slice */
  bslice_2d(C_comp, A_in, x, y, w, h, kw, kh);
  for (i = 0; i < 1; ++i) {
    sprintf(output_buf, output_msg, C_comp[i], C_actual[i], i);
    mu_assert(output_buf, C_comp[i] == C_actual[i]);
  }

  return 0;
}

static char* test_bslice_2d_3()
{
  uint8_t A_in[3] = {4,64,240};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_2d_3\nOutput Mismatch: \nComputed dim: %d, Actual dim: %d\n";
  int n_comp;
  int n_actual = 9;
  int x = 0;
  int y = 0;
  int w = 7;
  int h = 3;
  int kw = 3;
  int kh = 3;

  /* tests padding slice */
  n_comp = bslice_2d(C_comp, A_in, x, y, w, h, kw, kh);
  sprintf(output_buf, output_msg, n_comp, n_actual);
  mu_assert(output_buf, n_comp == n_actual);

  return 0;
}

static char* test_bslice_2d_4()
{
  uint8_t A_in[3] = {4,64,240};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_2d_4\nOutput Mismatch: \nComputed dim: %d, Actual dim: %d\n";
  int n_comp;
  int n_actual = 3;
  int x = -2;
  int y = 3;
  int w = 7;
  int h = 3;
  int kw = 3;
  int kh = 3;

  /* tests padding slice */
  n_comp = bslice_2d(C_comp, A_in, x, y, w, h, kw, kh);
  sprintf(output_buf, output_msg, n_comp, n_actual);
  mu_assert(output_buf, n_comp == n_actual);

  return 0;
}

static char* test_bslice_4d_1()
{
  uint8_t A_in[13] = {255,100,40,250,237,136,118,34,228,97,65,230,160};
  uint8_t C_actual[2] = {196,128};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_4d_1\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  int i;
  int x = 1;
  int y = 1;
  int zi = 0;
  int zj = 0;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;

  /* tests padding slice */
  bslice_4d(C_comp, A_in, x, y, zi, zj, w, h, d, kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, C_comp[i], C_actual[i], i);
    mu_assert(output_buf, C_comp[i] == C_actual[i]);
  }

  return 0;
}

static char* test_bslice_4d_2()
{
  uint8_t A_in[13] = {225,243,201,239,246,180,201,20,66,235,180,144,80};
  uint8_t C_actual[2] = {237,128};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_4d_2\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  int i;
  int x = 1;
  int y = 1;
  int zi = 0;
  int zj = 1;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;

  bslice_4d(C_comp, A_in, x, y, zi, zj, w, h, d,  kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, C_comp[i], C_actual[i], i);
    mu_assert(output_buf, C_comp[i] == C_actual[i]);
  }

  return 0;
}

static char* test_bslice_4d_3()
{
  uint8_t A_in[13] = {154,177,216,120,179,189,80,219,250,64,124,86,48};
  uint8_t C_actual[2] = {226,128};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_4d_3\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  int i;
  int x = 1;
  int y = 1;
  int zi = 1;
  int zj = 1;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;

  bslice_4d(C_comp, A_in, x, y, zi, zj, w, h, d,  kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, C_comp[i], C_actual[i], i);
    mu_assert(output_buf, C_comp[i] == C_actual[i]);
  }

  return 0;
}

static char* test_bslice_4d_4()
{
  uint8_t A_in[13] = {154,177,216,120,179,189,80,219,250,64,124,86,48};
  uint8_t C_comp[2] = {0};
  int n_actual = 9;

  char output_msg[] = "\nTEST: bslice_4d_4\nOutput Mismatch: \nComputed dim: %d, Actual dim: %d\n";
  int n_comp;
  int x = 0;
  int y = 0;
  int zi = 1;
  int zj = 1;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;

  n_comp = bslice_4d(C_comp, A_in, x, y, zi, zj, w, h, d,  kw, kh);
  sprintf(output_buf, output_msg, n_comp, n_actual);
  mu_assert(output_buf, n_comp == n_actual);

  return 0;
}

static char* test_bslice_4d_5()
{
  uint8_t A_in[13] = {154,177,216,120,179,189,80,219,250,64,124,86,48};
  uint8_t C_comp[2] = {0};
  int n_actual = 4;

  char output_msg[] = "\nTEST: bslice_4d_5\nOutput Mismatch: \nComputed dim: %d, Actual dim: %d\n";
  int n_comp;
  int x = -1;
  int y = -1;
  int zi = 1;
  int zj = 1;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;

  n_comp = bslice_4d(C_comp, A_in, x, y, zi, zj, w, h, d,  kw, kh);
  sprintf(output_buf, output_msg, n_comp, n_actual);
  mu_assert(output_buf, n_comp == n_actual);

  return 0;
}



static char* test_bdot_1()
{
  uint8_t A_in[3] = {1,175,248};
  uint8_t B_in[3] = {108,178,223};
  int actual = 1;
  int N = 21;
  int comp;
  char output_msg[] = "\nTEST: bdot_1\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  /* 21 len vector input */
  comp = bdot(A_in, B_in, N);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_2()
{
  uint8_t A_in[13] = {255,255,255,255,255,255,255,255,255,255,255,255,240};
  uint8_t B_in[13] = {255,255,255,255,255,255,255,255,255,255,255,255,255};
  int actual = 100;
  int N = 100;
  int comp;
  char output_msg[] = "\nTEST: bdot_2\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  /* 100 len vector input (all 1s) */
  comp = bdot(A_in, B_in, N);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3()
{
  uint8_t A_in[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
  uint8_t B_in[13] = {255,255,255,255,255,255,255,255,255,255,255,255,255};
  int actual = -100;
  int N = 100;
  int comp;
  char output_msg[] = "\nTEST: bdot_3\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  /* 100 len vector input (A -1s, B 1s) */
  comp = bdot(A_in, B_in, N);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3d_1()
{
  uint8_t A_in[16] = {43,18,84,186,190,191,119,232,93,74,174,215,87,182,21,168};
  uint8_t B_in[10] = {151,255,133,127,31,127,127,255,84,127};
  int actual = 1;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_1\nOutput Mismatch: \nComputed=%d, Actual=%d\n";
  int x = 0;
  int y = 0;
  int z = 0;
  int w = 5;
  int h = 5;
  int d = 5;
  int kw = 3;
  int kh = 3;
  comp = bdot_3d(A_in, B_in, x, y, z, w, h, d, kw, kh);

  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3d_2()
{
  uint8_t A_in[13] = {225,216,182,106,214,243,107,74,61,188,110,68,0};
  uint8_t B_in[8] = {240,255,35,127,19,255,136,255};
  int actual = 4;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_2\nOutput Mismatch: \nComputed=%d, Actual=%d\n";
  int x = 0;
  int y = 0;
  int z = 1;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  comp = bdot_3d(A_in, B_in, x, y, z, w, h, d, kw, kh);

  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3d_3()
{
  uint8_t A_in[13] = {1,20,31,95,8,203,26,71,79,178,160,245,160};
  uint8_t B_in[8] = {1,127,203,255,206,127,51,255};
  int actual = 2;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_3\nOutput Mismatch: \nComputed=%d, Actual=%d\n";
  int x = -1;
  int y = -1;
  int z = 1;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  comp = bdot_3d(A_in, B_in, x, y, z, w, h, d, kw, kh);

  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3d_4()
{
  uint8_t A_in[13] = {1,166,158,59,106,147,237,117,21,82,127,153,0};
  uint8_t B_in[8] = {53,127,221,255,216,255,171,255};
  int actual = 2;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_4\nOutput Mismatch: \nComputed=%d, Actual=%d\n";
  int x = 1;
  int y = 4;
  int z = 0;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  comp = bdot_3d(A_in, B_in, x, y, z, w, h, d, kw, kh);

  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_fdot_3d_1()
{
  float A_in[25] = {0.94678,0.9502,0.98926,0.95068,0.75146,0.51855,0.96777,0.26416,0.094543,0.30176,0.10272,0.76514,0.11639,0.86377,0.97998,0.67529,0.35132,0.71484,0.96924,0.066711,0.14062,0.69629,0.9834,0.44897,0.78711};
  uint8_t B_in[2] = {59, 127};
  float actual = -1.022705;
  float comp;
  char output_msg[] = "\nTEST: fdot_3d_1\nOutput Mismatch: \nComputed=%.3f, Actual=%.3f\n";
  int x = 1;
  int y = 1;
  int w = 5;
  int h = 5;
  int d = 1;
  int kw = 3;
  int kh = 3;

  comp = fdot_3d(A_in, B_in, x, y, w, h, d, kw, kh);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, fabs(comp - actual) < 1e-4);

  return 0;
}

static char* test_fdot_3d_2()
{
  float A_in[25] = {0.94678,0.9502,0.98926,0.95068,0.75146,0.51855,0.96777,0.26416,0.094543,0.30176,0.10272,0.76514,0.11639,0.86377,0.97998,0.67529,0.35132,0.71484,0.96924,0.066711,0.14062,0.69629,0.9834,0.44897,0.78711};
  uint8_t B_in[2] = {59, 127};
  float actual = -0.879635;
  float comp;
  char output_msg[] = "\nTEST: fdot_3d_2\nOutput Mismatch: \nComputed=%.3f, Actual=%.3f\n";
  int x = 3;
  int y = -1;
  int w = 5;
  int h = 5;
  int d = 1;
  int kw = 3;
  int kh = 3;

  comp = fdot_3d(A_in, B_in, x, y, w, h, d, kw, kh);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, fabs(comp - actual) < 1e-4);

  return 0;
}

static char* test_fdot_3d_3()
{
  float A_in[125] = {0.3064,0.66699,0.16321,0.48804,0.98633,0.93945,0.7002,0.77734,0.42139,0.85547,0.96387,0.085693,0.71387,0.076843,0.82812,0.73145,0.19641,0.47656,0.36255,0.83838,0.1521,0.83008,0.90625,0.3999,0.73389,0.2207,0.62549,0.6167,0.83691,0.050262,0.035004,0.57812,0.91162,0.046967,0.10205,0.93457,0.6709,0.82031,0.58057,0.3147,0.032532,0.56055,0.11414,0.18237,0.030197,0.21301,0.36426,0.7583,0.061523,0.77344,0.98877,0.16394,0.85449,0.37085,0.04245,0.2179,0.88721,0.56006,0.10461,0.54053,0.45825,0.25757,0.17896,0.81104,0.89844,0.97412,0.40576,0.29199,0.45581,0.11127,0.82227,0.036835,0.32715,0.31226,0.76318,0.35376,0.075073,0.13843,0.95947,0.79346,0.099121,0.29517,0.99658,0.22681,0.85889,0.24438,0.89795,0.026733,0.49683,0.94336,0.0017099,0.86133,0.73828,0.56201,0.57227,0.94775,0.17139,0.5791,0.45093,0.53271,0.89014,0.33374,0.4043,0.29199,0.28271,0.60791,0.79297,0.99219,0.21899,0.63379,0.5625,0.42383,0.44727,0.2218,0.56543,0.65674,0.8667,0.13428,0.53027,0.84814,0.16382,0.41455,0.34961,0.15332,0.42969};
  uint8_t B_in[6] = {217,240,190,109,135,23};
  float actual = 0.445281;
  float comp;
  char output_msg[] = "\nTEST: fdot_3d_3\nOutput Mismatch: \nComputed=%.3f, Actual=%.3f\n";
  int x = -1;
  int y = -1;
  int w = 5;
  int h = 5;
  int d = 5;
  int kw = 3;
  int kh = 3;

  comp = fdot_3d(A_in, B_in, x, y, w, h, d, kw, kh);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, fabs(comp - actual) < 1e-4);

  return 0;
}

static char* test_fconv_1()
{
  float A_in[100] = {0.2196,0.80322,0.52344,0.20178,0.89502,0.20825,0.85547,0.35864,0.56934,0.46558,0.44653,0.67871,0.80371,0.61133,0.71582,0.73047,0.80908,0.011681,0.59961,0.1969,0.087585,0.30371,0.57373,0.35352,0.11487,0.87012,0.75684,0.38135,0.95312,0.32471,0.82764,0.39404,0.92969,0.13708,0.73535,0.049194,0.68115,0.88379,0.80029,0.058533,0.88428,0.21521,0.21899,0.41943,0.12115,0.5,0.39697,0.15576,0.20361,0.16345,0.37231,0.6543,0.54053,0.74658,0.9873,0.064514,0.27979,0.62402,0.24426,0.65381,0.63721,0.5,0.23499,0.1825,0.50781,0.39697,0.28784,0.25244,0.41211,0.13623,0.7168,0.20093,0.89307,0.99316,0.32935,0.58398,0.56592,0.78662,0.30444,0.011703,0.05011,0.81592,0.65039,0.64502,0.74707,0.68311,0.49756,0.18518,0.78125,0.52783,0.1781,0.72949,0.15076,0.35571,0.5835,0.72363,0.65039,0.71826,0.83154,0.92236};
  uint8_t F_in[6] = {144,114,63,123,183,63};
  uint8_t C_actual[13] = {0,0,2,255,255,253,128,0,0,63,255,255,240};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_1\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i, res_size;

  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int c_idx = 0;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  res_size = fconv(A_in, F_in, C_comp, c_idx, bias, gamma, beta, mean, std,
                   w, h, d, kw, kh, sw, sh, pw, ph,
                   pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
  c_idx += res_size;

  for (i = 0; i < c_idx; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_2()
{
  float A_in[100] = {0.2196,0.80322,0.52344,0.20178,0.89502,0.20825,0.85547,0.35864,0.56934,0.46558,0.44653,0.67871,0.80371,0.61133,0.71582,0.73047,0.80908,0.011681,0.59961,0.1969,0.087585,0.30371,0.57373,0.35352,0.11487,0.87012,0.75684,0.38135,0.95312,0.32471,0.82764,0.39404,0.92969,0.13708,0.73535,0.049194,0.68115,0.88379,0.80029,0.058533,0.88428,0.21521,0.21899,0.41943,0.12115,0.5,0.39697,0.15576,0.20361,0.16345,0.37231,0.6543,0.54053,0.74658,0.9873,0.064514,0.27979,0.62402,0.24426,0.65381,0.63721,0.5,0.23499,0.1825,0.50781,0.39697,0.28784,0.25244,0.41211,0.13623,0.7168,0.20093,0.89307,0.99316,0.32935,0.58398,0.56592,0.78662,0.30444,0.011703,0.05011,0.81592,0.65039,0.64502,0.74707,0.68311,0.49756,0.18518,0.78125,0.52783,0.1781,0.72949,0.15076,0.35571,0.5835,0.72363,0.65039,0.71826,0.83154,0.92236};
  uint8_t F_in[6] = {144,114,63,123,183,63};
  uint8_t C_actual[13] = {0,0,2,255,255,253,128,0,0,63,255,255,240};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_2\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i, j, res_size, c_idx, a_idx, f_idx;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  c_idx = 0;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      a_idx = i*w*h*d;
      f_idx = j*(((kw*kh*d)/8)+1);
      res_size = fconv(A_in + a_idx, F_in + f_idx, C_comp, c_idx, bias, gamma,
                       beta, mean, std, w, h, d, kw, kh, sw, sh, pw, ph,
                       pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }

  for (i = 0; i < c_idx; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_1()
{
  uint8_t A_in[4] = {203,4,94,0};
  uint8_t F_in[2] = {129,127};
  uint8_t C_actual[2] = {239,0};
  uint8_t C_comp[4] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_1[No Padding]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i, res_size;

  int w = 5;
  int h = 5;
  int d = 1;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 0;
  int ph = 0;
  int c_idx = 0;
  int z = 0;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  res_size = bconv(A_in, F_in, C_comp, c_idx, z, bias, gamma, beta, mean, std, w, h, d,
                   kw, kh, sw, sh, pw, ph);
  c_idx += res_size;

  for (i = 0; i < c_idx; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_2()
{
  uint8_t A_in[4] = {190,2,145,0};
  uint8_t F_in[2] = {2,127};
  uint8_t C_actual[1] = {48};
  uint8_t C_comp[4] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_2[Stride=2]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i, res_size;

  int w = 5;
  int h = 5;
  int d = 1;
  int kw = 3;
  int kh = 3;
  int sw = 2;
  int sh = 2;
  int pw = 0;
  int ph = 0;
  int c_idx = 0;
  int z = 0;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  res_size = bconv(A_in, F_in, C_comp, c_idx, z, bias, gamma, beta, mean, std, w, h, d,
                   kw, kh, sw, sh, pw, ph);
  c_idx += res_size;

  for (i = 0; i < c_idx; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_1 () {
  float A_in[100] = {0.2196,0.80322,0.52344,0.20178,0.89502,0.20825,0.85547,0.35864,0.56934,0.46558,0.44653,0.67871,0.80371,0.61133,0.71582,0.73047,0.80908,0.011681,0.59961,0.1969,0.087585,0.30371,0.57373,0.35352,0.11487,0.87012,0.75684,0.38135,0.95312,0.32471,0.82764,0.39404,0.92969,0.13708,0.73535,0.049194,0.68115,0.88379,0.80029,0.058533,0.88428,0.21521,0.21899,0.41943,0.12115,0.5,0.39697,0.15576,0.20361,0.16345,0.37231,0.6543,0.54053,0.74658,0.9873,0.064514,0.27979,0.62402,0.24426,0.65381,0.63721,0.5,0.23499,0.1825,0.50781,0.39697,0.28784,0.25244,0.41211,0.13623,0.7168,0.20093,0.89307,0.99316,0.32935,0.58398,0.56592,0.78662,0.30444,0.011703,0.05011,0.81592,0.65039,0.64502,0.74707,0.68311,0.49756,0.18518,0.78125,0.52783,0.1781,0.72949,0.15076,0.35571,0.5835,0.72363,0.65039,0.71826,0.83154,0.92236};
  uint8_t F_in[6] = {144,114,63,123,183,63};
  uint8_t C_actual[13] = {0,0,2,255,255,253,128,0,0,63,255,255,240};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_1\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < m*d*w*h; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_2 () {
  float A_in[100] = {0.63818,0.91602,0.73193,0.59131,0.2959,0.50586,0.69434,0.47119,0.7002,0.99805,0.26172,0.87402,0.62744,0.47437,0.6875,0.84766,0.00098038,0.73389,0.59863,0.74316,0.46045,0.27075,0.68066,0.63428,0.94824,0.94336,0.56445,0.57031,0.23181,0.67236,0.06897,0.50537,0.89502,0.14514,0.7749,0.15686,0.69238,0.87939,0.69775,0.73926,0.41333,0.13928,0.39697,0.84277,0.52197,0.19092,0.5835,0.40259,0.56348,0.98828,0.98682,0.56934,0.47778,0.18909,0.64453,0.066467,0.74854,0.19458,0.45459,0.013565,0.31348,0.90625,0.82275,0.47583,0.54541,0.31689,0.59619,0.71973,0.21899,0.37305,0.048309,0.64844,0.7168,0.73877,0.071838,0.32129,0.65869,0.0053787,0.94727,0.2334,0.38721,0.12317,0.21667,0.42139,0.54785,0.9873,0.48608,0.67139,0.016769,0.037201,0.62109,0.70068,0.20288,0.80225,0.33325,0.93799,0.33374,0.57471,0.14197,0.26538};
  uint8_t F_in[6] = {38,215,63,232,245,127};
  uint8_t C_actual[5] = {53,255,216,191,112};
  uint8_t C_comp[5] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_2[Padding = 0]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 0;
  int ph = 0;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};


  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 36; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_3 () {
  float A_in[100] = {-0.1563336,0.20310879,0.45503479,0.10215032,-0.35107943,-0.47788608,0.43665612,-0.39295477,-0.242511,-0.16268927,0.24356085,-0.29996067,-0.36936045,0.32830894,0.30005246,-0.4229877,0.47766793,0.13914853,0.48771363,0.31482178,0.0056344867,0.031437397,0.2623691,0.18428946,0.29783833,-0.060771346,0.0074283481,-0.079036772,-0.16354576,-0.47341317,-0.28961819,0.34196794,-0.21572945,0.48987818,0.20053393,-0.24628094,0.085286736,0.011133015,0.15788889,-0.079722643,0.02951771,0.098571479,-0.089476377,0.38665992,0.41226524,0.31607103,-0.3019914,0.30643123,-0.24764237,0.02744621,0.23181123,-0.49995798,0.31550521,0.29372591,-0.46552521,0.056788027,0.26580203,-0.3501423,-0.23893347,-0.18649754,-0.2485832,-0.37679356,-0.22775337,-0.25505787,0.25351727,-0.45274761,-0.047463894,0.13269347,-0.23962706,0.39752251,0.085826039,-0.41205356,-0.34899443,0.21852303,-0.019280851,0.30807853,0.40730506,-0.12547112,-0.066619188,0.29639542,0.22854304,0.19678301,-0.46110496,0.2355094,0.11993122,0.098684788,0.1010595,-0.28327006,-0.4259575,0.15507746,0.22352308,0.22490931,0.012260854,0.0065851808,-0.4908233,-0.41289419,0.12294894,-0.30139416,-0.23328468,-0.38017982};
  uint8_t F_in[6] = {82,50,127,255,243,63};
  uint8_t C_actual[5] = {224,126,236,160,208};
  uint8_t C_comp[5] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_3[Stride = 2]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 2;
  int sh = 2;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;

  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};


  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 36; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_4 () {
  float A_in[100] = {0.27686,0.72656,0.99365,0.7251,0.032288,0.40308,0.11035,0.28467,0.15674,0.56543,0.030151,0.35767,0.86865,0.55322,0.11432,0.51221,0.65332,0.57031,0.78955,0.86279,0.68945,0.86816,0.65283,0.18079,0.73877,0.5498,0.57568,0.46973,0.99072,0.47363,0.1416,0.12158,0.22107,0.11414,0.98389,0.28833,0.96826,0.7334,0.97607,0.4375,0.0087051,0.044037,0.99609,0.64307,0.16382,0.1803,0.66748,0.43359,0.29736,0.55322,0.38306,0.065918,0.87402,0.067749,0.79053,0.30957,0.21143,0.21887,0.0495,0.093567,0.096069,0.84277,0.35376,0.59521,0.3313,0.51172,0.056824,0.875,0.13123,0.36499,0.26733,0.040802,0.31836,0.55371,0.10461,0.13452,0.47485,0.24243,0.44312,0.34863,0.19189,0.081848,0.67676,0.83154,0.67871,0.75977,0.92725,0.48926,0.4624,0.38647,0.14441,0.58252,0.38257,0.37524,0.99512,0.27148,0.90625,0.56445,0.14783,0.13086};
  uint8_t F_in[6] = {143,77,127,181,137,63};
  uint8_t C_actual[13] = {249,222,112,132,3,0,62,127,148,202,81,0,0};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_4[BatchNorm]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;

  int num_f = 2;
  float Bias[2] = {0.0040016,0.0060005};
  float Beta[2] = {-0.059998,0.010002};
  float Gamma[2] = {1.0195,1.2002};
  float Mean[2] = {0.065002,-0.11493};
  float Std[2] = {0.45776,0.35352};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 100; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_5 () {
  float A_in[100] = {0.024332345,0.22909111,0.44413209,-0.17742509,0.14547688,0.19454676,-0.19894373,0.18278843,-0.19984534,0.353755,0.090257764,0.15397388,-0.1891343,0.0021972656,0.39922035,0.43303382,-0.06565243,-0.19066125,0.23955423,-0.20480686,-0.037401944,0.38079107,0.39866644,0.31372589,0.41884011,0.37985432,0.060182929,0.17756248,0.16979343,0.29605931,0.29180133,0.092278481,0.4061783,-0.44684759,0.066815794,0.075803936,-0.25814646,0.41955054,-0.31753588,0.079267859,-0.11515993,0.46363163,0.21318763,-0.48170832,0.22276139,-0.48168308,0.14981711,0.47691792,0.10048044,0.023560405,-0.066285193,0.3869738,0.3833881,-0.45538226,0.30264854,-0.053524524,0.29490453,-0.49390617,0.086951613,0.30639493,0.23415482,0.22558755,-0.030328453,0.46945947,0.36584121,-0.23150504,-0.33181888,0.31827003,0.20722473,0.46441972,0.21648449,0.02259022,-0.49661499,0.23493934,0.46073931,0.29643756,0.41926169,-0.14935574,0.072524786,-0.41221991,-0.20521176,-0.23424631,0.20818347,-0.18881714,-0.43335968,-0.46128365,0.47022504,0.42814684,0.35564661,0.45685875,0.2465362,0.12842578,-0.49647173,0.15315545,0.2998336,0.087063551,0.48379707,-0.17894232,-0.084107608,0.30229038};
  uint8_t F_in[6] = {144,142,127,208,228,127};
  uint8_t C_actual[2] = {127,255};
  uint8_t C_comp[2] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_4[Pooling (No Padding/No Stride)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 2;
  int pl_h = 2;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;

  int num_f = 2;
  float Bias[2] = {0.0040016,0.0060005};
  float Beta[2] = {-0.059998,0.010002};
  float Gamma[2] = {1.0195,1.2002};
  float Mean[2] = {0.065002,-0.11493};
  float Std[2] = {0.45776,0.35352};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 100; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_layer_1 () {
  uint8_t A_in[13] = {208,40,250,231,111,235,200,200,19,101,206,21,112};
  uint8_t F_in[8] = {233,127,191,127,236,255,78,127};
  uint8_t C_actual[5] = {107,9,46,26,64};
  uint8_t C_comp[5] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_1[No Padding]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 0;
  int ph = 0;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 36; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_layer_2 () {
  uint8_t A_in[13] = {185,207,250,198,200,236,93,17,10,0,234,82,0};
  uint8_t F_in[8] = {132,127,238,255,175,127,213,255};
  uint8_t C_actual[13] = {78,49,5,153,63,243,253,63,124,57,152,136,80};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_2[Padding]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 100; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_layer_3 () {
  uint8_t A_in[13] = {15,0,155,77,186,149,242,50,86,95,97,7,64};
  uint8_t F_in[8] = {91,127,22,255,181,127,160,127};
  uint8_t C_actual[5] = {204,226,237,163,176};
  uint8_t C_comp[5] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_3[Stride=2]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 2;
  int sh = 2;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 36; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}


static char* test_bconv_layer_4 () {
  uint8_t A_in[13] = {145,159,201,124,94,132,216,167,24,114,216,30,112};
  uint8_t F_in[8] = {123,127,220,127,136,255,13,127};
  uint8_t C_actual[13] = {216,221,172,120,27,13,193,176,64,246,210,63,176};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_4[BatchNorm]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int num_f = 2;
  float Bias[2] = {0.0040016,0.0060005};
  float Beta[2] = {-0.059998,0.010002};
  float Gamma[2] = {1.0195,1.2002};
  float Mean[2] = {0.028397,-0.019394};
  float Std[2] = {1.4287,1.3418};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 100; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}



static char* test_blinear_layer_1 () {
  uint8_t A_in[160] = {223,220,2,167,118,210,29,66,80,32,248,230,79,133,233,108,226,38,93,239,148,166,71,102,133,201,170,225,249,8,123,150,238,127,198,121,100,99,169,49,228,11,19,35,92,28,17,236,103,118,81,217,61,212,58,225,64,184,52,214,53,58,112,160,211,73,81,190,39,151,177,164,76,34,209,218,69,88,78,58,216,35,83,82,12,147,96,51,60,57,125,142,177,172,192,142,245,240,35,106,73,66,157,242,112,149,222,79,66,172,97,232,63,78,8,112,5,236,190,211,253,158,211,118,76,45,209,136,214,111,139,45,45,182,0,70,6,107,238,144,87,61,70,100,4,95,33,75,94,94,61,115,179,47,37,193,155,234,109,108};
  uint8_t F_in[32] = {233,6,115,180,129,72,93,153,182,51,73,224,18,65,218,53,92,10,252,156,0,207,189,160,46,156,24,180,243,31,127,159};
  uint8_t C_actual[3] = {145,243,0};
  uint8_t C_comp[3] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: blinear_layer_1\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 10;
  int n = 127;
  int k = 2;
  float Bias[2] = {0.0040016,0.0060005};
  float Beta[2] = {-0.059998,0.010002};
  float Gamma[2] = {1.0195,1.2002};
  float Mean[2] = {-0.37964,-0.1394};
  float Std[2] = {1.7686,3.0664};


  blinear_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, n, k);

  for (i = 0; i < 20; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}


static char* all_tests()
{
  mu_run_test(test_bslice_2d_1);
  mu_run_test(test_bslice_2d_2);
  mu_run_test(test_bslice_2d_3);
  mu_run_test(test_bslice_2d_4);
  mu_run_test(test_bslice_4d_1);
  mu_run_test(test_bslice_4d_2);
  mu_run_test(test_bslice_4d_3);
  mu_run_test(test_bslice_4d_4);
  mu_run_test(test_bslice_4d_5);
  mu_run_test(test_bdot_1);
  mu_run_test(test_bdot_2);
  mu_run_test(test_bdot_3);
  mu_run_test(test_bdot_3d_1);
  mu_run_test(test_bdot_3d_2);
  mu_run_test(test_bdot_3d_3);
  mu_run_test(test_bdot_3d_4);
  mu_run_test(test_fdot_3d_1);
  mu_run_test(test_fdot_3d_2);
  mu_run_test(test_fdot_3d_3);
  /* mu_run_test(test_fconv_1); */
  /* mu_run_test(test_fconv_2); */
  mu_run_test(test_bconv_1);
  mu_run_test(test_bconv_2);
  /* mu_run_test(test_fconv_layer_1); */
  /* mu_run_test(test_fconv_layer_2); */
  /* mu_run_test(test_fconv_layer_3); */
  /* mu_run_test(test_fconv_layer_4); */
  mu_run_test(test_fconv_layer_5);
  mu_run_test(test_bconv_layer_1);
  mu_run_test(test_bconv_layer_2);
  mu_run_test(test_bconv_layer_3);
  mu_run_test(test_bconv_layer_4);
  mu_run_test(test_blinear_layer_1);
  return 0;
}

int main ()
{
  char *result = all_tests();
  if (result != 0) {
    printf("%s\n", result);
  }
  else {
    printf("ALL TESTS PASSED\n");
  }
  printf("Tests run: %d\n", tests_run);

  return 0;
}
