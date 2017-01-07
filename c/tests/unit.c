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

static char* test_bslice_3d_1()
{
  uint8_t A_in[7] = {172,8,238,207,80,123,128};
  uint8_t C_actual[2] = {163,128};
  uint8_t C_comp[2] = {0};

  char output_msg[] = "\nTEST: bslice_3d_1\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  int i;
  int x = 1;
  int y = 1;
  int z = 1;
  int w = 5;
  int h = 5;
  int kw = 3;
  int kh = 3;

  /* tests padding slice */
  bslice_3d(C_comp, A_in, x, y, z, w, h, kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, C_comp[i], C_actual[i], i);
    mu_assert(output_buf, C_comp[i] == C_actual[i]);
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
  uint8_t A_in[16] = {43,18,84,186,190,191,119,232,93,74,174,215,87,182,21,168};
  uint8_t B_in[10] = {151,255,133,127,31,127,127,255,84,127};
  int actual = 1;
  int comp;
  char output_msg[] = "\nTEST: bdot_3d_1\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  comp = bdot_3d(A_in, B_in, 0, 0, 5, 5, 5, 3, 3);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}

static char* test_bdot_3d_2()
{
  uint8_t A_in[16] = {120,106,152,142,59,84,11,54,43,107,15,165,209,194,214,136};
  uint8_t B_in[10] = {52,127,160,127,92,127,83,255,182,127};
  int actual = 1;
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
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};


  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f, w,
              h, d, kw, kh, sw, sh, pw, ph);

  for (i = 0; i < 36; ++i) {
    actual = nthbitset_arr(C_actual, i);
    comp = nthbitset_arr(C_comp, i);
    sprintf(output_buf, output_msg, i, comp, i, actual);
    mu_assert(output_buf, comp == actual);
  }

  return 0;
}

static char* test_fconv_layer_3 () {
  float A_in[100] = {0.6748,0.59033,0.055023,0.80127,0.03363,0.93848,0.14893,0.16602,0.58545,0.29883,0.57275,0.32373,0.34033,0.056213,0.89307,0.52051,0.22046,0.94287,0.31812,0.81201,0.29517,0.5127,0.43481,0.61084,0.45996,0.75195,0.50488,0.24329,0.49878,0.70947,0.9043,0.47363,0.57861,0.87598,0.75049,0.58643,0.027084,0.57129,0.071228,0.604,0.83057,0.66602,0.59766,0.22693,0.74707,0.88623,0.57471,0.77197,0.58154,0.65723,0.2384,0.44995,0.26831,0.77295,0.54395,0.84863,0.52783,0.26489,0.6499,0.41504,0.24121,0.81494,0.59326,0.72656,0.36084,0.8623,0.83545,0.59717,0.17468,0.15137,0.39868,0.83057,0.56006,0.76807,0.17041,0.95703,0.22192,0.96582,0.68359,0.059113,0.74561,0.066833,0.19067,0.75,0.060425,0.5625,0.62061,0.34644,0.14856,0.63867,0.28027,0.58936,0.60107,0.061462,0.29297,0.65771,0.97607,0.037628,0.92822,0.11273};
  uint8_t F_in[6] = {64,189,255,34,121,255};
  uint8_t C_actual[5] = {122,148,36,37,0};
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
  int num_f = 2;
  float Bias[2] = {0.0, 0.0};
  float Gamma[2] = {1.0, 1.0};
  float Beta[2] = {0.0, 0.0};
  float Mean[2] = {0.0, 0.0};
  float Std[2] = {1.0, 1.0};


  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f, w,
              h, d, kw, kh, sw, sh, pw, ph);

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
  int num_f = 2;
  float Bias[2] = {0.0040016,0.0060005};
  float Beta[2] = {-0.059998,0.010002};
  float Gamma[2] = {1.0195,1.2002};
  float Mean[2] = {0.065002,-0.11493};
  float Std[2] = {0.45776,0.35352};

  fconv_layer(A_in, F_in, C_comp, Bias, Gamma, Beta, Mean, Std, m, num_f, w,
              h, d, kw, kh, sw, sh, pw, ph);

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
  mu_run_test(test_bslice_3d_1);
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
  mu_run_test(test_fconv_layer_2);
  mu_run_test(test_fconv_layer_3);
  mu_run_test(test_fconv_layer_4);
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
