#include <stdio.h>
#include <math.h>
#include "../ebnn.h"
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
  int i;

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

  fconv(A_in, F_in, C_comp, c_idx, bias, gamma, beta, mean, std,
        w, h, d, kw, kh, sw, sh, pw, ph,
        pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

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
  int i;

  int w = 5;
  int h = 5;
  int d = 1;
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
  int c_idx = 0;
  int z = 0;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  bconv(A_in, F_in, C_comp, c_idx, z, bias, gamma, beta, mean, std, w, h, d,
        kw, kh, sw, sh, pw, ph, pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

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
  int i;

  int w = 5;
  int h = 5;
  int d = 1;
  int kw = 3;
  int kh = 3;
  int sw = 2;
  int sh = 2;
  int pw = 0;
  int ph = 0;
  int pl_w = 1;
  int pl_h = 1;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 0;
  int pl_ph = 0;
  int c_idx = 0;
  int z = 0;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  bconv(A_in, F_in, C_comp, c_idx, z, bias, gamma, beta, mean, std, w, h, d,
        kw, kh, sw, sh, pw, ph, pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < c_idx; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_1 () {
  float A_in[100] = {-0.33977616,-0.38752186,0.095303118,-0.019698232,-0.1167936,-0.38064414,-0.08819741,0.18753332,-0.20444921,-0.3050257,0.34070009,0.40679836,-0.28664404,-0.28581673,-0.12480527,-0.14833218,-0.016386062,-0.080009103,-0.42706889,0.2558195,0.009604454,-0.32294518,-0.22238797,-0.46135914,0.20100993,0.091738701,0.10914958,-0.027439475,0.38335657,-0.20047429,-0.23634997,0.10986,0.40964973,-0.39354566,-0.23903432,-0.10409451,-0.10784316,0.44449639,0.40569448,-0.40030512,0.39817017,0.077887058,-0.37817982,-0.025002539,0.041511238,0.012455583,-0.055420101,-0.43717578,-0.086635321,-0.45515648,-0.05630371,-0.3397516,-0.44387609,0.069186568,-0.40823272,0.43693137,0.10810214,0.14043581,0.24327976,0.30313224,-0.47718668,0.036825478,-0.48936209,0.062210441,-0.34581712,0.11792701,-0.036171407,-0.19858599,-0.35721254,0.059763908,0.41031706,-0.32694355,0.09653312,-0.081225216,-0.45775032,-0.16524762,-0.038872629,0.20563304,-0.30958992,-0.15654984,0.044617057,-0.36743337,-0.41760647,0.17152655,-0.10049835,-0.053191155,0.39964336,0.31699252,0.49975938,0.33872098,-0.17459905,0.083599746,0.19851238,-0.20699185,0.32760435,-0.36998969,-0.20145991,-0.038289398,0.26987314,-0.29047567};
  uint8_t F_in[6] = {26,111,63,33,49,191};
  uint8_t C_actual[13] = {179,156,108,59,233,55,209,85,124,219,120,45,240};
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

  for (i = 0; i < 100; ++i) {
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
  float A_in[100] = {-0.11535534,-0.1723671,-0.1348764,-0.17158738,0.29882151,-0.33514118,-0.086955875,-0.025106847,0.049130559,-0.13267654,-0.02254656,0.13413215,-0.28580084,0.1066466,-0.10726413,0.40748221,0.36703038,-0.18321344,0.45956302,-0.30952907,0.39161754,0.41614413,0.055625916,0.36360252,0.098271668,-0.44520465,-0.18953016,0.26392913,0.41106665,0.23892391,0.29233313,0.43424618,0.055956781,-0.22045273,0.18135738,0.044909716,0.19357407,0.37603074,0.025488436,-0.30130339,0.18767071,-0.2361528,0.28118432,0.16629946,-0.25912404,-0.15410414,0.2034018,0.030583918,-0.21443042,0.27263027,0.24890894,-0.2300247,-0.23544747,0.19678503,0.080870152,-0.47772983,0.10618538,-0.15190715,-0.21256825,-0.062744409,-0.1647425,-0.45580795,-0.48416331,-0.091655314,-0.1442247,-0.42628229,0.054817796,-0.12852633,0.15928113,-0.0097763836,-0.38916123,-0.17960674,0.35079014,-0.30582255,0.065004528,-0.094003618,0.36026061,0.47209835,0.056161761,0.49408185,0.050706565,0.067509949,0.26862222,0.078600883,0.010173023,-0.094081819,-0.21475407,0.12641907,0.26222551,0.21609509,-0.15863493,-0.0018675029,-0.2523905,0.20307773,-0.11539358,-0.054499716,0.37792188,-0.35523802,-0.10334882,-0.19549301};
  uint8_t F_in[6] = {52,73,127,62,244,191};
  uint8_t C_actual[8] = {191,119,127,255,255,255,255,255};
  uint8_t C_comp[2] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_5[Pooling (No Padding/Stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 64; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_6 () {
  float A_in[100] = {0.16352028,0.061680853,0.12245911,0.011493981,0.424941,0.33885068,0.13441551,0.38981164,-0.49826998,-0.30067098,-0.42576671,-0.25077969,-0.10274065,0.35476243,-0.19905528,0.25782382,0.22519886,-0.40619743,-0.051692545,-0.39655954,0.36964196,-0.12455568,0.12432355,0.22673446,-0.43149999,-0.21319875,0.36145526,-0.10521597,-0.009428978,0.30392933,-0.024185807,-0.39688861,0.48563451,-0.11788607,-0.058764249,0.36762959,0.23734057,-0.038861424,0.074200928,-0.17177424,0.42401427,0.19193238,-0.21737,0.13799387,-0.27427262,-0.034839809,0.21829033,0.091215193,0.078286648,0.42266333,-0.38552979,-0.23001224,-0.29691675,0.23396701,0.12464023,-0.33556634,-0.33898538,0.36903691,-0.46530348,-0.3390356,-0.15943801,0.41931289,0.28238881,0.47125244,-0.15787333,0.34230304,-0.44806421,0.1071164,-0.01523909,0.13506722,-0.084840477,-0.0036567152,-0.11060107,0.0026220083,-0.34263176,-0.042930305,0.16221416,0.12064034,0.302975,0.39819437,-0.43394125,-0.071971476,-0.45984504,0.27629846,-0.21001768,0.36251098,0.33615446,-0.38502288,-0.18012694,0.064185679,-0.28836459,-0.30574656,-0.12942451,-0.49602252,-0.42011589,0.15612286,-0.31126124,0.42883372,0.41252661,0.3044675};
  uint8_t F_in[6] = {33,130,255,82,201,255};
  uint8_t C_actual[18] = {243,237,255,60,119,191,255,247,223,115,239,223,125,240,51,253,239,247};
  uint8_t C_comp[18] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_6[Pooling (Padding=1/Stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  int pl_pw = 1;
  int pl_ph = 1;

  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 144; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_7 () {
  float A_in[144] = {-0.38699165,-0.11422881,0.17732465,0.41427422,0.28321117,-0.24860767,0.38613659,-0.16823062,0.42593396,-0.031929374,0.42988974,-0.23219857,0.24866956,0.28080434,-0.48488572,-0.22550681,-0.15219685,-0.014407486,-0.22922426,0.38030696,0.34052086,-0.11492401,0.34856266,0.034075797,-0.48393944,-0.16265357,0.44021529,-0.069621742,-0.48266095,0.3551777,-0.14002499,-0.11540118,-0.24717224,-0.26210064,-0.46489927,0.4516955,-0.4571383,-0.46168527,-0.34896433,0.32379258,-0.23042548,-0.34623796,-0.36970055,0.36836141,0.017956138,0.38165891,-0.013893574,0.096223891,-0.11830705,-0.3111971,-0.30811834,0.27996558,-0.0044978559,-0.36909223,0.27342057,0.34976536,0.289621,-0.37642902,-0.3204599,-0.035168022,0.15286225,-0.081231743,-0.4851453,0.38836622,-0.20252082,-0.42969882,0.33948153,0.28432447,0.3803671,-0.11425456,-0.076933354,0.42203331,-0.16334611,0.44812179,0.099147022,0.25315124,0.33473653,0.30393648,0.205971,0.0378021,0.27763045,-0.42561769,-0.28622556,0.12715214,0.079083204,-0.20478436,-0.48284295,-0.22915331,-0.13452199,0.4926756,-0.47524351,-0.47667927,-0.20365861,-0.16208091,-0.47740391,0.38842809,-0.43875647,-0.46590012,-0.10864624,0.34003502,0.32538319,-0.068750083,-0.036323875,0.10992712,-0.44338542,0.074074447,0.26930279,0.31199396,-0.30917937,-0.37690881,0.084756315,0.18131274,-0.19174421,-0.28468552,-0.45624664,-0.37825197,-0.27412963,-0.16880819,-0.014868319,-0.47178614,0.18921947,-0.20373622,0.0832569,-0.014576226,0.11591452,0.099219501,-0.014504254,-0.44923037,-0.36633676,-0.19473025,-0.23725569,0.30682606,0.34519321,-0.41842681,0.12973207,0.069204867,-0.17466307,-0.028610021,0.015362382,0.49023873,0.25962287,0.16176611,-0.098460227,0.22100449};
  uint8_t F_in[6] = {22,174,255,19,60,191};
  uint8_t C_actual[8] = {119,238,183,238,34,252,102,252};
  uint8_t C_comp[8] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_7[Pooling (Padding=1/Stride=2)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 6;
  int h = 6;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 2;
  int pl_h = 2;
  int pl_sw = 2;
  int pl_sh = 2;
  int pl_pw = 1;
  int pl_ph = 1;

  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 64; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_8 () {
  float A_in[100] = {0.18095076,-0.10406068,-0.13514516,0.15204841,-0.071808219,-0.29753008,0.20571142,-0.21908721,-0.43688914,0.19843632,-0.4317258,-0.078213155,0.028377831,0.092931688,-0.44711772,-0.09963876,-0.46497008,-0.30300972,0.16816431,0.012782335,-0.33802038,-0.11081856,-0.48804048,-0.14878929,0.41953164,-0.48315167,-0.11160651,-0.26039141,-0.36467963,0.17901659,-0.07832554,-0.20116326,-0.074266106,-0.011144876,-0.26586667,0.49653429,-0.092710793,0.27780306,0.48137605,0.037166893,-0.17198709,-0.17069393,-0.044468552,-0.27879632,-0.34542012,0.28560227,-0.33260512,0.45142972,0.29911977,0.0258919,-0.45499119,-0.053697377,-0.49095422,-0.400639,0.12189245,0.047791481,-0.34277657,-0.10517678,0.31575781,0.26208419,0.071237564,-0.3571915,0.46412688,0.49694705,-0.24761096,-0.38792777,0.43677992,-0.093469441,-0.36233544,-0.086604834,-0.11642766,-0.15038636,0.09849,0.25421882,-0.44482598,-0.11660522,0.28444111,0.38045162,0.077434242,-0.0030651391,-0.20743904,-0.29068384,-0.26708674,-0.0470182,-0.31514233,-0.26453859,-0.10904995,0.4455744,0.28392202,0.29713631,0.048559427,0.38722777,0.21626431,0.33886617,-0.021876663,-0.23926294,-0.42186207,0.27412945,0.34524149,-0.42246425};
  uint8_t F_in[6] = {143,239,63,121,153,63};
  uint8_t C_actual[13] = {27,223,255,255,255,57,255,255,255,255,255,255,240};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_8[Pooling (poolsize=3)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  int pl_w = 3;
  int pl_h = 3;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 1;
  int pl_ph = 1;

  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

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

static char* test_fconv_layer_9 () {
  float A_in[100] = {-0.35315156,0.21253908,0.24908596,-0.42286554,-0.41587767,-0.14968938,0.28284866,0.47159392,-0.20732945,-0.070013285,-0.09845987,-0.33470601,0.35890174,-0.46163175,0.45585084,-0.058127224,0.47709411,0.044495344,0.43149287,-0.0018969774,-0.22963473,0.13746333,0.13737476,-0.07884267,-0.25367057,-0.14670691,0.12152016,-0.13843048,-0.29554477,-0.38831261,0.39594841,-0.23742089,-0.25468051,0.085982919,-0.42049676,-0.28377235,0.26524532,-0.48411876,-0.0067997575,-0.49285313,0.023470938,-0.17451298,-0.043434173,0.18832862,0.059429288,0.02578491,0.048388243,-0.2803497,0.052647114,0.41633189,0.13186222,0.46301901,-0.45992935,0.25857574,-0.2362808,0.4590804,0.35928905,-0.42495617,0.24117154,-0.28950551,0.41600657,-0.2846933,0.090892375,-0.35335481,-0.33488131,0.45233297,0.24103975,-0.45690641,0.48238742,-0.014910191,-0.29572481,-0.37768254,-0.12248865,0.1091069,-0.48792282,-0.37126684,-0.43130663,-0.3988834,-0.40073633,0.12779725,0.075774014,0.014054418,-0.27989519,0.40212083,-0.0093907118,-0.053876519,-0.38946968,0.12919849,-0.24252921,0.22573435,-0.40550208,-0.024394184,0.34711307,-0.17801115,-0.41787589,0.014821112,-0.44885215,0.10926986,0.46567708,-0.45113292};
  uint8_t F_in[6] = {222,235,63,193,19,255};
  uint8_t C_actual[2] = {168,165};
  uint8_t C_comp[2] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_9[Conv(stride=3) + No Pooling]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 3;
  int sh = 3;
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

  for (i = 0; i < 16; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_10 () {
  float A_in[100] = {0.26793754,0.066765904,0.2247389,-0.23467788,-0.2506156,0.21934092,-0.0091305077,-0.12005308,0.41994876,-0.036086798,0.070175409,-0.081526577,0.41396904,0.4585948,0.23853922,-0.065385014,-0.013491929,0.41950166,-0.47502175,0.33852839,0.21489483,0.47154033,0.20353806,-0.21739233,0.31404966,0.27584541,0.45527726,-0.019888759,0.098291755,0.38700294,-0.45877281,0.11309481,0.035027921,-0.26322323,-0.18279302,-0.27617618,-0.18671584,-0.17733771,0.1660983,-0.4871403,-0.34892476,-0.43841106,-0.018259108,-0.077148318,0.21191627,-0.42809865,0.047302365,0.28082305,-0.36760294,0.26467031,0.0045985579,0.0056532621,0.34601885,-0.46684241,-0.43856603,0.29614812,0.45490593,-0.23485491,-0.29696411,-0.43807507,0.4954266,0.16603559,0.48595828,-0.036981761,-0.22037998,0.26259261,0.18522024,-0.28289175,-0.15146953,-0.20764196,0.040085971,-0.016490608,-0.45715716,0.36635619,0.062144339,0.16188771,-0.095196605,0.30480826,-0.25835901,-0.44462901,0.47085571,0.12487304,-0.35169375,-0.28034064,-0.22789896,0.047939718,-0.40459806,0.47173208,-0.13500321,0.37697095,-0.03300786,-0.13970518,-0.42636672,-0.091722965,0.048245609,0.41743523,0.33295184,0.052132964,0.32505828,0.038994789};
  uint8_t F_in[6] = {204,183,63,43,159,191};
  uint8_t C_actual[5] = {221,254,3,123,0};
  uint8_t C_comp[5] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_9[Conv(stride=3) + Pooling]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 2;
  int w = 5;
  int h = 5;
  int d = 2;
  int kw = 3;
  int kh = 3;
  int sw = 3;
  int sh = 3;
  int pw = 1;
  int ph = 1;
  int pl_w = 2;
  int pl_h = 2;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 1;
  int pl_ph = 1;

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

static char* test_fconv_layer_11 () {
  float A_in[100] = {-0.11952704,-0.31531614,-0.35320619,-0.50417829,-0.21558017,-0.52651262,-0.37631193,0.13156325,0.12863517,-0.77228177,-0.16753799,-0.27951396,-0.22845799,-0.26229388,-0.53081679,-0.61870074,-0.33026734,0.068819046,-0.67967546,-0.64691961,-0.71531487,-0.78872216,-0.33562189,-0.7696957,-0.39072049,-0.66992939,-0.66981411,-0.06056881,-0.19180459,-0.15514934,-0.12909335,0.036578,0.078361869,-0.46866688,-0.39557487,0.15391499,-0.21120781,-0.34638056,-0.52948856,-0.53630471,-0.013397753,-0.60043037,0.17713702,0.015891314,-0.61019552,0.12196022,0.074864388,-0.3832033,-0.13582957,0.12659556,-0.51298642,0.063358784,-0.083692849,-0.6751985,0.10680205,-0.034693837,-0.41329628,-0.038058698,-0.58020729,-0.14997286,-0.20193374,-0.25310254,-0.56817991,0.044179201,-0.59818268,-0.30977935,-0.058521211,-0.49868944,-0.39677808,0.087112546,-0.31074372,-0.71540415,-0.0069015026,-0.55020559,-0.15357405,-0.23044169,-0.57516301,-0.59939688,-0.11524677,-0.062312186,-0.084714711,-0.58850628,-0.57670575,-0.44633353,0.061677217,-0.66776896,-0.5508697,-0.23156154,0.016060352,-0.73234147,-0.42582837,-0.34067515,0.045278668,0.052455723,-0.16877264,-0.1567747,-0.59212422,-0.55113387,0.091227353,-0.54461741};
  uint8_t F_in[6] = {167,91,255,126,156,191};
  uint8_t C_actual[13] = {255,255,255,187,222,192,0,13,255,248,193,206,112};
  uint8_t C_comp[13] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_11[Conv(stride=1) + Pooling(stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  int pl_w = 3;
  int pl_h = 3;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 1;
  int pl_ph = 1;

  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

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

static char* test_fconv_layer_12 () {
  float A_in[36] = {-0.56463647,-0.29729176,-0.42165029,-0.21281511,-0.14583987,-0.65500516,-0.71884739,-0.275392,-0.85040152,-0.24986833,-0.55124295,-0.65826952,-0.58335054,-0.25083023,-0.18527454,-0.44041735,-0.0060428381,-0.34078628,-0.47037348,-0.37082875,-0.77611804,-0.21166229,-0.70595747,-0.3613503,-0.13939512,-0.44365406,-0.29199547,-0.13939166,-0.66095078,-0.39832067,-0.80822074,-0.55792636,-0.67169166,-0.38358527,-0.7747916,-0.2148596};
  uint8_t F_in[4] = {249,255,152,127};
  uint8_t C_actual[9] = {0,0,0,0,15,255,255,255,255};
  uint8_t C_comp[9] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: fconv_layer_12[Conv(stride=1) + Pooling(stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 1;
  int w = 6;
  int h = 6;
  int d = 1;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 3;
  int pl_h = 3;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 1;
  int pl_ph = 1;

  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 72; ++i) {
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

static char* test_bconv_layer_5 () {
  uint8_t A_in[13] = {109,125,86,190,223,65,109,117,7,36,211,5,16};
  uint8_t F_in[8] = {220,255,87,255,32,127,37,255};
  uint8_t C_actual[8] = {255,255,236,159,255,191,223,255};
  uint8_t C_comp[8] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_5[Pooling (Padding=0, Stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 64; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_layer_6 () {
  uint8_t A_in[13] = {22,239,98,37,60,128,218,13,69,43,12,29,112};
  uint8_t F_in[8] = {147,127,237,127,73,255,44,127};
  uint8_t C_actual[18] = {239,190,255,255,127,255,255,255,207,251,255,223,251,231,255,255,255,255};
  uint8_t C_comp[18] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_6[Pooling (Padding=1, Stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  int pl_pw = 1;
  int pl_ph = 1;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 144; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_layer_7 () {
  uint8_t A_in[13] = {151,196,21,137,214,209,24,198,246,221,187,140,80};
  uint8_t F_in[8] = {244,127,10,255,19,127,42,127};
  uint8_t C_actual[8] = {63,247,255,255,255,255,119,119};
  uint8_t C_comp[18] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_7[Conv (Stride=2), Pooling (Padding=1, Stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
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
  int pl_w = 2;
  int pl_h = 2;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 1;
  int pl_ph = 1;
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 64; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_bconv_layer_8 () {
  uint8_t A_in[9] = {4,0,5,16,0,18,128,64,0};
  uint8_t F_in[8] = {217,127,93,255,72,255,2,127};
  uint8_t C_actual[9] = {195,143,190,56,239,255,255,255,255};
  uint8_t C_comp[9] = {0};
  uint8_t comp, actual;
  char output_msg[] = "\nTEST: bconv_layer_8[Conv (Stride=1), Pooling (Padding=1, Stride=1)]\nOutput Mismatch: \nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  int m = 1;
  int d = 2;
  int w = 6;
  int h = 6;
  int num_f = 2;
  int kw = 3;
  int kh = 3;
  int sw = 1;
  int sh = 1;
  int pw = 1;
  int ph = 1;
  int pl_w = 3;
  int pl_h = 3;
  int pl_sw = 1;
  int pl_sh = 1;
  int pl_pw = 1;
  int pl_ph = 1;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};

  bconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f,
              w, h, d, kw, kh, sw, sh, pw, ph,
              pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);

  for (i = 0; i < 72; ++i) {
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
  //TODO: refactor test since we take the max after linear

  /* for (i = 0; i < 20; ++i) { */
  /*   actual = nthbitset_arr(C_actual, i); */
  /*   comp = nthbitset_arr(C_comp, i); */
  /*   sprintf(output_buf, output_msg, i, comp, i, actual); */
  /*   mu_assert(output_buf, comp == actual); */
  /* } */

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
  mu_run_test(test_fconv_1);
  mu_run_test(test_bconv_1);
  mu_run_test(test_bconv_2);
  mu_run_test(test_fconv_layer_1);
  mu_run_test(test_fconv_layer_2);
  mu_run_test(test_fconv_layer_3);
  mu_run_test(test_fconv_layer_4);
  mu_run_test(test_fconv_layer_5);
  mu_run_test(test_fconv_layer_6);
  mu_run_test(test_fconv_layer_7);
  mu_run_test(test_fconv_layer_8);
  mu_run_test(test_fconv_layer_9);
  mu_run_test(test_fconv_layer_10);
  mu_run_test(test_fconv_layer_11);
  mu_run_test(test_fconv_layer_12);
  mu_run_test(test_bconv_layer_1);
  mu_run_test(test_bconv_layer_2);
  mu_run_test(test_bconv_layer_3);
  mu_run_test(test_bconv_layer_4);
  mu_run_test(test_bconv_layer_5);
  mu_run_test(test_bconv_layer_6);
  /* mu_run_test(test_bconv_layer_7); */
  mu_run_test(test_bconv_layer_8);
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
