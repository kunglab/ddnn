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

  int i;
  int kw, kh;
  char output_msg[] = "\nTEST: bslice_2d_1\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  kw = kh = 3;

  /* tests normal slice */
  bslice_2d(slice_2d_comp, slice_2d_in, 0, 4, 7, 3, kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, slice_2d_out[i], slice_2d_comp[i], i);
    mu_assert(output_buf, slice_2d_out[i] == slice_2d_comp[i]);
  }

  return 0;
}

static char* test_bslice_2d_2()
{
  uint8_t slice_2d_in[3] = {223,175,248};
  uint8_t slice_2d_out[2] = {3,128};
  uint8_t slice_2d_comp[2] = {0};

  int i;
  int kw, kh;
  char output_msg[] = "\nTEST: bslice_2d_2\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  kw = kh = 3;

  /* tests padding slice */
  bslice_2d(slice_2d_comp, slice_2d_in, -2, 3, 7, 3, kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, slice_2d_out[i], slice_2d_comp[i], i);
    mu_assert(output_buf, slice_2d_out[i] == slice_2d_comp[i]);
  }

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
  uint8_t A_in[20] = {255,117,211,255,75,161,124,127,18,162,60,255,158,195,244,127,124,246,18,127};
  uint8_t B_in[10] = {178,255,209,255,20,255,108,127,80,127};
  int actual = 13;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_1\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  comp = bdot_3d(A_in, B_in, 1, 2, 5, 5, 5, 3, 3);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3d_2()
{
  uint8_t A_in[20] = {255,117,211,255,75,161,124,127,18,162,60,255,158,195,244,127,124,246,18,127};
  uint8_t B_in[10] = {178,255,209,255,20,255,108,127,80,127};
  int actual = -1;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_2\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  comp = bdot_3d(A_in, B_in, -1, -1, 5, 5, 5, 3, 3);
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
  int c_idx = 0;
  float bias = 0.0;
  float gamma = 1.0;
  float beta = 0.0;
  float mean = 0.0;
  float std = 1.0;

  res_size = fconv(A_in, F_in, C_comp, c_idx, bias, gamma, beta, mean, std, w, h, d,
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
                       beta, mean, std, w, h, d, kw, kh, sw, sh, pw, ph);
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
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};


  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f, w,
              h, d, kw, kh, sw, sh, pw, ph);

  for (i = 0; i < m*d*w*h; ++i) {
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
  mu_run_test(test_bdot_1);
  mu_run_test(test_bdot_2);
  mu_run_test(test_bdot_3);
  mu_run_test(test_bdot_3d_1);
  mu_run_test(test_bdot_3d_2);
  mu_run_test(test_fdot_3d_1);
  mu_run_test(test_fdot_3d_2);
  mu_run_test(test_fdot_3d_3);
  mu_run_test(test_fconv_1);
  mu_run_test(test_fconv_2);
  mu_run_test(test_fconv_layer_1);
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
