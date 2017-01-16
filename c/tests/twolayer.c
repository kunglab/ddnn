#include <stdio.h>
#include "minunit.h"
#include "twolayer.h"
#include "twolayer_compare.h"

int tests_run = 0;
char output_buf[1000];
int errors = 0;

static char* test_l1() {
  char output_msg[] = "Output Mismatch: Layer 1.\nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  l_conv_pool_bn_bst0(x_in, temp1);

  for (i = 0; i < 196; ++i) {
    sprintf(output_buf, output_msg, i, temp1[i], i, inter1[i]);
    mu_assert(output_buf, temp1[i] == inter1[i]);
  }

  return 0;
}

static char* test_l2() {
  char output_msg[] = "Output Mismatch: Layer 2.\nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;

  l_b_conv_pool_bn_bst1(inter1, temp1);

  for (i = 0; i < 196; ++i) {
    sprintf(output_buf, output_msg, i, temp1[i], i, inter2[i]);
    mu_assert(output_buf, temp1[i] == inter2[i]);
  }

  return 0;
}

static char* test_l3() {
  char output_msg[] = "Output Mismatch: Layer 3.\nComputed: %d, Actual: %d\n";
  uint8_t output[1];

  blinear_layer(inter2, l_b_linear_bn_softmax2_bl_W, output,
                l_b_linear_bn_softmax2_bl_b, l_b_linear_bn_softmax2_bn_gamma,
                l_b_linear_bn_softmax2_bn_beta, l_b_linear_bn_softmax2_bn_mean,
                l_b_linear_bn_softmax2_bn_std, 1, 1568, 10);

  sprintf(output_buf, output_msg, output[0], 7);
  mu_assert(output_buf,  output[0] == 7);

  return 0;
}

static char* test_endtoend() {
  char output_msg[] = "Output Mismatch: Layer 3.\nComputed: %d, Actual: %d\n";
  uint8_t output[1];

  ebnn_compute(x_in, output);

  sprintf(output_buf, output_msg, output[0], 7);
  mu_assert(output_buf,  output[0] == 7);

  return 0;
}

static char* all_tests() {
  mu_run_test(test_l1);
  mu_run_test(test_l2);
  mu_run_test(test_l3);
  mu_run_test(test_endtoend);
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
