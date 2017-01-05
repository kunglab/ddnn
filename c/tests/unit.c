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


static char* test_binary_dot() {
  char output_msg[] = "\nTEST: bdot\nOutput Mismatch: \nComputed=%d, Actual=%d at index (%d,%d)\n";

  return 0;
}


static char* all_tests() {
  mu_run_test(test_float_dot);
  mu_run_test(test_binary_dot);
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
