#include <stdio.h>
#include "../util.h"
#include "minunit.h"
#include "unit_compare.h"
#define W 28
#define H 28

int tests_run = 0;
char output_buf[1000];
int errors = 0;

static char* test_float_dot() {
  int i, j, out_idx;
  int kw, kh, pw, ph, in_chans;
  uint8_t res_sign, act_res_sign;
  float res;
  char output_msg[] = "\nTEST: fdot\nOutput Mismatch: \nComputed=%d, Actual=%d at index (%d,%d)\n";

  in_chans = 1;
  kw = kh = 3;
  ph = pw = 1;
  out_idx = 0;
  for (i = -pw; i < W-kw+pw+1; ++i) {
    for (j = -ph; j < H-kh+ph+1; ++j) {
      res = fdot(float_in, W1, in_chans, i, j, W, H, kw, kh);
      res += Bias1[0];
      res = BN(res, Gamma1[0], Beta1[0], Mean1[0], Std1[0]);
      res_sign = res > 0 ? 1 : 0;
      act_res_sign = nthbitset_arr(fdot_out, out_idx);
      out_idx++;

      sprintf(output_buf, output_msg, res_sign, act_res_sign, i, j);
      mu_assert(output_buf,  res_sign == act_res_sign);
    }
  }

  return 0;
}


static char* test_slice_2d_1()
{
  uint8_t slice_2d_in[3] = {223,175,248};
  uint8_t slice_2d_out[2] = {239,128};
  uint8_t slice_2d_comp[2] = {0};

  int i;
  int kw, kh;
  char output_msg[] = "\nTEST: slice_2d\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  kw = kh = 3;

  /* tests normal slice */
  slice_2d(slice_2d_comp, slice_2d_in, 0, 4, 7, 3, kw, kh);
  for (i = 0; i < 2; ++i) {
    sprintf(output_buf, output_msg, slice_2d_out[i], slice_2d_comp[i], i);
    mu_assert(output_buf, slice_2d_out[i] == slice_2d_comp[i]);
  }

  return 0;
}

static char* test_slice_2d_2()
{
  uint8_t slice_2d_in[3] = {223,175,248};
  uint8_t slice_2d_out[2] = {3,128};
  uint8_t slice_2d_comp[2] = {0};

  int i;
  int kw, kh;
  char output_msg[] = "\nTEST: slice_2d\nOutput Mismatch: \nComputed=%d, Actual=%d at index %d\n";
  kw = kh = 3;

  /* tests padding slice */
  slice_2d(slice_2d_comp, slice_2d_in, -2, 3, 7, 3, kw, kh);
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
  char output_msg[] = "\nTEST: bdot\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

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
  char output_msg[] = "\nTEST: bdot\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

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
  char output_msg[] = "\nTEST: bdot\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

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
  char output_msg[] = "\nTEST: bdot_3d\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

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
  char output_msg[] = "\nTEST: bdot_3d\nOutput Mismatch: \nComputed=%d, Actual=%d\n";

  comp = bdot_3d(A_in, B_in, -1, -1, 5, 5, 5, 3, 3);
  sprintf(output_buf, output_msg, comp, actual);
  mu_assert(output_buf, comp == actual);

  return 0;
}


static char* all_tests() {
  mu_run_test(test_slice_2d_1);
  mu_run_test(test_slice_2d_2);
  mu_run_test(test_bdot_1);
  mu_run_test(test_bdot_2);
  mu_run_test(test_bdot_3);
  mu_run_test(test_bdot_3d_1);
  mu_run_test(test_bdot_3d_2);
  mu_run_test(test_float_dot);
  return 0;
}

int main () {
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
